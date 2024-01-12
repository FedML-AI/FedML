from typing import Any, Dict, Optional, Sized, Union

from collections import OrderedDict
import gc
import logging
import math
import os
from pathlib import Path
import warnings

from accelerate.utils import broadcast_object_list
from datasets import IterableDataset
import fedml
from fedml import FedMLRunner, mlops
from fedml.arguments import Arguments
from fedml.core import ClientTrainer, ServerAggregator
from fedml.train.llm.configurations import DatasetArguments, ExperimentArguments, ModelArguments
from fedml.train.llm.distributed import barrier
from fedml.train.llm.hf_trainer import HFTrainer
from fedml.train.llm.modeling_utils import get_data_collator, to_device
from fedml.train.llm.train_utils import (
    get_dataset,
    get_model,
    get_max_seq_length,
    get_tokenizer,
)
from fedml.train.llm.typing import DatasetType, ModelType, PathType, TokenizerType
from fedml.train.llm.utils import (
    get_real_path,
    is_file,
    parse_hf_args,
    save_config,
)
from peft import PeftModel
from peft.utils import WEIGHTS_NAME as PEFT_WEIGHTS_NAME
import torch.cuda
from torch.nn import Module
from transformers import PreTrainedModel
from transformers.utils import WEIGHTS_NAME as HF_WEIGHTS_NAME

from src.fedllm_trainer import FedLLMTrainer
from src.peft_utils import set_peft_model_state_dict
from src.trainer_callback import PauseResumeCallback
from src.utils import log_helper


def _parse_args(args: Arguments) -> Arguments:
    # When launched with `torchrun`, local_rank is set via environment variable
    # see https://pytorch.org/docs/stable/elastic/run.html
    args.local_rank = int(os.getenv("LOCAL_RANK", args.local_rank))

    if not hasattr(args, "output_dir"):
        raise ValueError("`output_dir` is required in the configuration file.")

    args.output_dir = get_real_path(args.output_dir.format(run_id=args.run_id))
    args.output_dir = str(Path(args.output_dir) / f"node_{args.rank}")

    # set default value
    if not hasattr(args, "is_aggregator_test"):
        args.is_aggregator_test = False
    if not hasattr(args, "use_customized_hierarchical"):
        args.use_customized_hierarchical = False
    if not hasattr(args, "test_on_client_ranks"):
        args.test_on_client_ranks = []
    if not hasattr(args, "test_on_clients"):
        args.test_on_clients = "no"
    if not hasattr(args, "local_num_train_epochs"):
        args.local_num_train_epochs = None
    if not hasattr(args, "local_max_steps"):
        args.local_max_steps = None
    if not hasattr(args, "frequency_of_the_test"):
        args.frequency_of_the_test = 1
    if not hasattr(args, "save_frequency"):
        args.save_frequency = None

    # verify and update
    if args.role == "client":
        if hasattr(args, "client_dataset_path"):
            args.dataset_path = args.client_dataset_path

        if args.test_on_clients == "no" or args.rank not in args.test_on_client_ranks:
            # disable huggingface Trainer's logging when not testing on client
            args.report_to = "none"
            args.disable_tqdm = True

    if hasattr(args, "client_dataset_path"):
        delattr(args, "client_dataset_path")

    if isinstance(args.dataset_path, str):
        args.dataset_path = [args.dataset_path]

    if isinstance(args.dataset_path, (tuple, list)):
        args.dataset_path = [
            get_real_path(p.format(rank=args.rank, client_num_in_total=args.client_num_in_total))
            for p in args.dataset_path
        ]

    if torch.cuda.device_count() == 0:
        logging.warning(f"{args.role} rank {args.rank} does not have GPU! Fallback to CPU mode.")
        args.deepspeed = None
        args.use_flash_attention = False

    # set default value for `num_train_epochs` and `local_num_train_epochs`
    if (
            getattr(args, "num_train_epochs", None) is not None and
            args.local_num_train_epochs is None
    ):
        # if `num_train_epochs` is present but not `local_num_train_epochs`
        warnings.warn(
            "`num_train_epochs` is deprecated and will be removed in future version. Use "
            "`local_num_train_epochs` instead.",
            FutureWarning
        )
        args.local_num_train_epochs = args.num_train_epochs

    if args.local_num_train_epochs is None:
        # set to HF default value
        args.local_num_train_epochs = 3.0

    # set default value for `max_steps` and `local_max_steps`
    if (
            getattr(args, "max_steps", None) is not None and
            args.local_max_steps is None
    ):
        # if `max_steps` is present but not `local_max_steps`
        warnings.warn(
            "`max_steps` is deprecated and will be removed in future version. Use "
            "`local_max_steps` instead.",
            FutureWarning
        )
        args.local_max_steps = args.max_steps

    if args.local_max_steps is None:
        # set to HF default value
        args.local_max_steps = -1

    assert args.local_max_steps > 0 or args.local_num_train_epochs > 0, \
        f"At least 1 of `local_max_steps` and `local_num_train_epochs` should be positive, " \
        f"but got {args.local_max_steps} and {args.local_num_train_epochs}"

    # update `num_train_epochs` and `max_steps`
    args.num_train_epochs = args.local_num_train_epochs * args.comm_round
    args.max_steps = args.local_max_steps * args.comm_round

    # update save and evaluation settings
    if args.save_frequency is None or args.save_frequency < 0:
        args.save_frequency = args.frequency_of_the_test

    return args


def _save_checkpoint(
        model: Module,
        checkpoint_dir: PathType,
        state_dict: Optional[Dict[str, Any]] = None
) -> None:
    if state_dict is None:
        state_dict = model.state_dict()

    if isinstance(model, (PeftModel, PreTrainedModel)):
        model.save_pretrained(
            save_directory=str(checkpoint_dir),
            state_dict=state_dict
        )
    else:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, str(checkpoint_dir / HF_WEIGHTS_NAME))


def save_checkpoint(
        model_or_trainer: Union[HFTrainer, Module],
        checkpoint_dir: Optional[PathType] = None,
        is_saving_process: Optional[bool] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        synchronize: bool = True
) -> None:
    """
    Save model checkpoint to a directory.

    Args:
        model_or_trainer: model or HF Trainer to save.
        checkpoint_dir: output directory for the checkpoint.
        is_saving_process: Whether the current process is the saving process. When this function is called on
            multiple processes, set `is_saving_process=True` only on the main process to avoid race conditions.
        state_dict: state_dict object to save. This overrides `model_or_trainer`.
        synchronize: Whether to synchronize after saving the model. This is required if you want to load the
            checkpoint immediately after saving.

    Returns:

    """
    # This function need to be called on all processes
    if isinstance(model_or_trainer, HFTrainer):
        if checkpoint_dir is None:
            checkpoint_dir = model_or_trainer.args.output_dir
        if is_saving_process is None:
            is_saving_process = model_or_trainer.args.should_save

    # verify args
    if checkpoint_dir is None:
        raise ValueError(
            f"`checkpoint_dir` is required for `model_or_trainer` type \"{type(model_or_trainer)}\"."
        )
    if is_saving_process is None:
        raise ValueError(
            f"`is_saving_process` is required for `model_or_trainer` type"
            f" \"{type(model_or_trainer)}\"."
        )

    # save model checkpoint
    if isinstance(model_or_trainer, HFTrainer):
        model_or_trainer.save_checkpoint(checkpoint_dir)

    elif isinstance(model_or_trainer, Module):
        if is_saving_process:
            _save_checkpoint(model_or_trainer, checkpoint_dir, state_dict)

    else:
        raise TypeError(f"\"{type(model_or_trainer)}\" is not a supported type.")

    if synchronize:
        # all process should wait
        barrier()


def load_checkpoint(checkpoint_dir: PathType) -> OrderedDict:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir / HF_WEIGHTS_NAME
    peft_checkpoint_path = checkpoint_dir / PEFT_WEIGHTS_NAME

    # TODO: support HF sharded checkpoints, see `transformers.utils.WEIGHTS_INDEX_NAME`
    if is_file(peft_checkpoint_path):
        state_dict = torch.load(str(peft_checkpoint_path), map_location="cpu")
    elif is_file(checkpoint_path):
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    else:
        raise FileNotFoundError(
            f"Could not find either PEFT checkpoint in \"{peft_checkpoint_path}\" nor full checkpoint"
            f" in {checkpoint_path}."
        )

    return state_dict


class LLMTrainer(ClientTrainer):
    def __init__(
            self,
            model: ModelType,
            args: Arguments,
            tokenizer: TokenizerType,
            training_args: ExperimentArguments,
            model_args: ModelArguments,
            dataset_args: DatasetArguments
    ):
        super().__init__(model, args)

        self.tokenizer = tokenizer
        self.training_args = training_args
        self.model_args = model_args
        self.dataset_args = dataset_args
        self.trainer = FedLLMTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            data_collator=get_data_collator(
                tokenizer=self.tokenizer,
                response_template=self.dataset_args.response_template,
                # set to `pad_to_multiple_of` to max_seq_length so that all distributed processes share the same
                # sequence length. This is required for computing metrics.
                pad_to_multiple_of=self.dataset_args.max_seq_length
            )
        )

        step_threshold = self.args.local_max_steps
        epoch_threshold = self.args.local_num_train_epochs

        if step_threshold > 0:
            epoch_threshold = math.inf
        elif epoch_threshold > 0:
            step_threshold = math.inf
        else:
            raise ValueError(
                f"At least 1 of `local_max_steps` and `local_num_train_epochs` should be positive, "
                f"but got {step_threshold} and {epoch_threshold}"
            )
        self.log(f"step_threshold = {step_threshold}, epoch_threshold = {epoch_threshold}")

        self.trainer.add_callback(PauseResumeCallback(
            step_threshold=step_threshold,
            epoch_threshold=epoch_threshold
        ))

        # save config
        if self.training_args.should_save:
            # save model config before training
            save_config(model, self.checkpoint_dir / "final")

        # track latest checkpoint
        self.latest_checkpoint_dir = self.checkpoint_dir / "init"

        barrier()
        self.log("initialized")

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.trainer.args.output_dir)

    def is_main_process(self) -> bool:
        return self.trainer.is_world_process_zero()

    def log(self, message: Any, stack_level: int = 1) -> None:
        log_helper(
            message,
            prefix=f"{{{{rank={self.args.rank}, world_rank={self.trainer.args.process_index}, "
                   f"local_rank={self.args.local_rank}, hf_local_rank={self.trainer.args.local_process_index}}}}}",
            suffix=f"@ round={self.round_idx}",
            stack_prefix=f"{type(self).__name__}.",
            stack_level=stack_level + 1
        )

    def get_model_params(self) -> OrderedDict:
        self.log("start")

        peft_state_dict = load_checkpoint(self.latest_checkpoint_dir)

        self.log("finished")
        return OrderedDict(peft_state_dict)

    def set_model_params(self, model_parameters) -> None:
        self.log("start")

        model_parameters = to_device(model_parameters, device="cpu")

        barrier()
        set_peft_model_state_dict(self.model, model_parameters)
        barrier()

        if self.round_idx >= 0 and self.should_save:
            # save aggregated model checkpoint
            self.latest_checkpoint_dir = self.checkpoint_dir / f"round_{self.round_idx}_after_agg"
            self.log(f"saving aggregated model to \"{self.latest_checkpoint_dir}\"")
            save_checkpoint(
                self.model,
                self.latest_checkpoint_dir,
                is_saving_process=self.training_args.should_save,
                state_dict=model_parameters,
                synchronize=True
            )

        self.log("finished")

    def on_before_local_training(self, train_data, device, args: Arguments) -> None:
        self.log("start")

        outputs = super().on_before_local_training(train_data, device, args)

        if isinstance(train_data, IterableDataset):
            # When streaming dataset, need to manually skip data samples
            num_sample_to_skip = self.trainer.state.global_step * self.training_args.train_batch_size
            self.trainer.train_dataset = train_data.skip(num_sample_to_skip)

            self.log(f"Skip first {num_sample_to_skip:,} samples for iterable train dataset")
        else:
            self.trainer.train_dataset = train_data

        self.log("finished")
        return outputs

    def train(self, train_data, device, args: Arguments) -> None:
        self.log("start")

        self.trainer.train()

        self.log("finished")

    def on_after_local_training(self, train_data, device, args: Arguments) -> None:
        self.log("start")

        outputs = super().on_after_local_training(train_data, device, args)

        self.latest_checkpoint_dir = self.checkpoint_dir / f"round_{self.round_idx}_before_agg"
        self.log(f"saving model to \"{self.latest_checkpoint_dir}\"")
        save_checkpoint(self.trainer, self.latest_checkpoint_dir)

        self.log("finished")
        return outputs

    def test(self, test_data, device, args) -> None:
        self.log("start")

        if not self.should_evaluate:
            self.log("skipped")
            return

        if len(self.args.test_on_client_ranks) <= 1:
            # use default prefix if only test on no more than 1 client
            metric_key_prefix = "eval"
        else:
            metric_key_prefix = f"client{self.args.rank}_eval"

        metrics = self.trainer.evaluate(eval_dataset=test_data, metric_key_prefix=metric_key_prefix)
        if self.is_main_process():
            mlops.log({**metrics, "round_idx": self.round_idx})

        self.log("finished")

    @property
    def should_evaluate(self) -> bool:
        return (
                self.args.test_on_clients != "no" and
                self.args.rank in self.args.test_on_client_ranks and
                # TODO: remove once `fedml` supports it
                (
                        self.round_idx % self.args.frequency_of_the_test == 0 or
                        self.round_idx == self.args.comm_round - 1
                )
        )

    @property
    def should_save(self) -> bool:
        if self.args.save_frequency is None or self.args.save_frequency < 0:
            return self.should_evaluate
        else:
            return self.round_idx % self.args.save_frequency == 0 or self.round_idx == self.args.comm_round - 1

    @property
    def round_idx(self) -> int:
        return getattr(self.args, "round_idx", -1)

    @round_idx.setter
    def round_idx(self, round_idx: int) -> None:
        self.args.round_idx = round_idx

    def sync_process_group(
            self,
            round_idx: Optional[int] = None,
            model_params: Optional[Any] = None,
            client_index: Optional[int] = None,
            from_process: int = 0
    ) -> None:
        self.log("start")

        if round_idx is None:
            round_idx = self.round_idx

        broadcast_object_list([round_idx, model_params, client_index], from_process=from_process)

        self.log("finished")

    def await_sync_process_group(self, from_process: int = 0) -> list:
        self.log("start")

        outputs = broadcast_object_list([None, None, None], from_process=from_process)

        self.log("finished")
        return outputs


class LLMAggregator(ServerAggregator):
    def __init__(
            self,
            model: ModelType,
            args: Arguments,
            tokenizer: TokenizerType,
            training_args: ExperimentArguments,
            model_args: ModelArguments,
            dataset_args: DatasetArguments
    ):
        super().__init__(model, args)

        self.tokenizer = tokenizer
        self.model_args = model_args
        self.dataset_args = dataset_args
        self.training_args = training_args
        self.trainer = FedLLMTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            data_collator=get_data_collator(
                tokenizer=self.tokenizer,
                response_template=self.dataset_args.response_template,
                # set to `pad_to_multiple_of` to max_seq_length so that all distributed processes share the same
                # sequence length. This is required for computing metrics.
                pad_to_multiple_of=self.dataset_args.max_seq_length
            )
        )

        # save config
        if self.training_args.should_save:
            # save model config before training
            save_config(model, self.checkpoint_dir / "final")

        self.latest_checkpoint_dir = self.checkpoint_dir / "init"

        barrier()
        self.log("initialized")

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.trainer.args.output_dir)

    def is_main_process(self) -> bool:
        return self.trainer.is_world_process_zero()

    def log(self, message: Any, stack_level: int = 1) -> None:
        log_helper(
            message,
            prefix=f"{{{{rank={self.args.rank}, world_rank={self.trainer.args.process_index}, "
                   f"local_rank={self.args.local_rank}, hf_local_rank={self.trainer.args.local_process_index}}}}}",
            suffix=f"@ round={self.round_idx}",
            stack_prefix=f"{type(self).__name__}.",
            stack_level=stack_level + 1
        )

    def get_model_params(self) -> OrderedDict:
        self.log("start")

        peft_state_dict = load_checkpoint(self.latest_checkpoint_dir)

        self.log("finished")
        return OrderedDict(peft_state_dict)

    def set_model_params(self, model_parameters) -> None:
        self.log("start")

        model_parameters = to_device(model_parameters, device="cpu")

        barrier()
        set_peft_model_state_dict(self.model, model_parameters)
        barrier()

        if self.round_idx >= 0 and self.should_save:
            # save aggregated model checkpoint
            self.latest_checkpoint_dir = self.checkpoint_dir / f"round_{self.round_idx}_after_agg"
            self.log(f"saving aggregated model to \"{self.latest_checkpoint_dir}\"")
            save_checkpoint(
                self.model,
                self.latest_checkpoint_dir,
                is_saving_process=self.training_args.should_save,
                state_dict=model_parameters,
                synchronize=True
            )

        self.log("finished")

    def test(self, test_data, device, args: Arguments) -> None:
        self.log("start")

        if not self.should_evaluate:
            self.log("skipped")
            return

        # update epoch, global_step for logging
        self.trainer.state.epoch = self.round_idx
        self.trainer.state.global_step = self.round_idx
        metrics = self.trainer.evaluate(eval_dataset=test_data)
        if self.is_main_process():
            mlops.log({**metrics, "round_idx": self.round_idx})

        self.log("finished")

    @property
    def should_evaluate(self) -> bool:
        return self.args.is_aggregator_test

    @property
    def should_save(self) -> bool:
        if self.args.save_frequency is None or self.args.save_frequency < 0:
            return self.should_evaluate
        else:
            return self.round_idx % self.args.save_frequency == 0 or self.round_idx == self.args.comm_round - 1

    @property
    def round_idx(self) -> int:
        return getattr(self.args, "round_idx", -1)

    @round_idx.setter
    def round_idx(self, round_idx: int) -> None:
        self.args.round_idx = round_idx


def transform_data_to_fedml_format(
        args: Arguments,
        training_args: ExperimentArguments,
        dataset_args: DatasetArguments,
        train_dataset: DatasetType,
        test_dataset: DatasetType
):
    if isinstance(train_dataset, Sized):
        train_data_num = len(train_dataset)
    elif training_args.max_steps > 0:
        # interpolate dataset size
        train_data_num = training_args.max_steps * training_args.train_batch_size * training_args.world_size
    else:
        raise ValueError(
            f"`local_max_steps` must be set to a positive value if dataloader does not have a length"
            f" (e.g., dataset streaming), was {args.local_max_steps}"
        )

    if isinstance(test_dataset, Sized):
        test_data_num = len(test_dataset)
    elif dataset_args.eval_dataset_size > 0:
        test_data_num = dataset_args.eval_dataset_size
    else:
        raise ValueError(
            f"`eval_dataset_size` must be set to a positive value if dataloader does not have a length"
            f" (e.g., dataset streaming), was {dataset_args.eval_dataset_size}"
        )

    train_data_global = None
    test_data_global = None
    train_data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    if args.rank == 0:
        # server data
        test_data_global = test_dataset
    else:
        # client data
        train_data_local_num_dict[args.rank - 1] = train_data_num
        train_data_local_dict[args.rank - 1] = train_dataset
        test_data_local_dict[args.rank - 1] = test_dataset
    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        2  # num classes, this is ignored for FedLLM
    )


def main(args: Arguments) -> None:
    # init device
    device = fedml.device.get_device(args)

    model_args, dataset_args = parse_hf_args((ModelArguments, DatasetArguments), args)

    if args.role == "server" and args.local_rank == 0:
        # Initialize model before initializing TrainingArgs to load the full model in memory
        # This is required when using DeepSpeed Zero3
        model = get_model(
            model_args,
            tokenizer_length=len(get_tokenizer(model_args)),
            use_cache=not getattr(args, "gradient_checkpointing", False)
        )

        # save initial model. This is required for DeepSpeed Zero3
        save_checkpoint(
            model_or_trainer=model,
            checkpoint_dir=Path(args.output_dir) / "init",
            is_saving_process=True,
            synchronize=False  # do not synchronize here
        )
        del model
        gc.collect()
    barrier()

    training_args, *_ = parse_hf_args(ExperimentArguments, args)
    # verify and update configs
    training_args.add_and_verify_args(model_args, dataset_args)

    # update cross-silo hierarchical related settings
    if args.use_customized_hierarchical:
        args.proc_rank_in_silo = training_args.process_index
        args.rank_in_node = training_args.local_process_index
        args.process_id = training_args.process_index

    # tokenizer need to be recreated after `transformers.TrainingArguments` to avoid serialization problems
    tokenizer = get_tokenizer(model_args)

    model = get_model(
        model_args,
        tokenizer_length=len(tokenizer),
        use_cache=not training_args.gradient_checkpointing
    )

    if dataset_args.max_seq_length is None:
        dataset_args.max_seq_length = get_max_seq_length(model)
        args.max_seq_length = dataset_args.max_seq_length

    # load data
    with training_args.main_process_first(local=True):
        train_dataset, _, test_dataset = get_dataset(
            dataset_args=dataset_args,
            tokenizer=tokenizer,
            seed=training_args.seed,
            is_local_main_process=training_args.local_process_index == 0
        )

        # prepend current rank to the seed then shuffle the training set
        # this is required for geo-distributed training
        train_dataset = train_dataset.shuffle(seed=int(f"{args.rank}{training_args.seed}"))

    dataset = transform_data_to_fedml_format(args, training_args, dataset_args, train_dataset, test_dataset)

    # FedML trainer
    trainer = aggregator = None
    if args.role == "client":
        trainer = LLMTrainer(
            model=model,
            args=args,
            tokenizer=tokenizer,
            training_args=training_args,
            model_args=model_args,
            dataset_args=dataset_args
        )
    elif args.role == "server":
        aggregator = LLMAggregator(
            model=model,
            args=args,
            tokenizer=tokenizer,
            training_args=training_args,
            model_args=model_args,
            dataset_args=dataset_args
        )
    else:
        raise RuntimeError(f"Invalid value for \"role\". Only \"client\" and \"server\" "
                           f"are allowed but received \"{args.role}\"")

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()


if __name__ == "__main__":
    # init FedML framework
    main(args=_parse_args(fedml.init()))
