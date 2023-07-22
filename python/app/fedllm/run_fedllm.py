from typing import Any, Optional, Union

from collections import OrderedDict
import gc
import logging
import math
from pathlib import Path

from accelerate.utils import broadcast_object_list
from datasets import Dataset
import fedml
from fedml import FedMLRunner, mlops
from fedml.arguments import Arguments
from fedml.core import ClientTrainer, ServerAggregator
from peft import get_peft_model_state_dict, PeftModel
from peft.utils import WEIGHTS_NAME as PEFT_WEIGHTS_NAME
import torch.cuda
from torch.nn import Module
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import WEIGHTS_NAME as HF_WEIGHTS_NAME

from run_train import (
    DatasetArguments,
    FinetuningArguments,
    get_dataset,
    get_model,
    get_max_seq_length,
    get_tokenizer,
    ModelArguments,
    ModelType,
    SavePeftModelCallback,
    TokenizerType,
)
from src.constants import DEFAULT_MAX_SEQ_LENGTH, RESPONSE_KEY_NL
from src.hf_resume_trainer import HFResumeTrainer
from src.hf_trainer import HFTrainer
from src.modeling_utils import get_data_collator
from src.peft_utils import set_peft_model_state_dict
from src.trainer_callback import PauseResumeCallback
from src.typing import PathType
from src.utils import (
    barrier,
    get_real_path,
    is_deepspeed_module,
    is_file,
    is_main_process,
    log_helper,
    parse_hf_args,
    save_config,
    should_process_save,
    to_device,
)


def _parse_args(args: Arguments) -> Arguments:
    if args.role == "client":
        if hasattr(args, "client_dataset_path"):
            args.dataset_path = args.client_dataset_path
        if not getattr(args, "is_client_test", False):
            # disable huggingface Trainer's logging when not testing on client
            setattr(args, "report_to", "none")
        setattr(args, "disable_tqdm", True)

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
        setattr(args, "deepspeed", None)

    if not hasattr(args, "output_dir"):
        raise ValueError("\"output_dir\" is required in the configuration file.")

    args.output_dir = get_real_path(args.output_dir.format(run_id=args.run_id))
    args.output_dir = str(Path(args.output_dir) / f"node_{args.rank}")

    return args


def get_hf_trainer(
        args: Arguments,
        model: ModelType,
        tokenizer: TokenizerType,
        training_args: FinetuningArguments,
        **kwargs
) -> HFResumeTrainer:
    return HFResumeTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=get_data_collator(
            tokenizer,
            escape_token=RESPONSE_KEY_NL if training_args.is_instruction_finetune else None,
            pad_to_multiple_of=getattr(args, "max_seq_length", DEFAULT_MAX_SEQ_LENGTH)
        ),
        **kwargs
    )


def save_model_helper(model: Module, checkpoint_dir: PathType) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / HF_WEIGHTS_NAME
    peft_checkpoint_path = checkpoint_dir / PEFT_WEIGHTS_NAME

    state_dict = to_device(model.state_dict(), device="cpu")

    if isinstance(model, PeftModel):
        torch.save(
            get_peft_model_state_dict(model, state_dict=state_dict),
            str(peft_checkpoint_path)
        )
    else:
        torch.save(state_dict, str(checkpoint_path))

    del state_dict
    gc.collect()


def save_model_state_dict(
        model_or_trainer: Union[HFTrainer, Module],
        checkpoint_dir: Optional[PathType] = None,
        is_saving_process: Optional[bool] = None
) -> None:
    if isinstance(model_or_trainer, HFTrainer):
        if checkpoint_dir is None:
            checkpoint_dir = model_or_trainer.args.output_dir
        if is_saving_process is None:
            is_saving_process = should_process_save(model_or_trainer)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / HF_WEIGHTS_NAME
    peft_checkpoint_path = checkpoint_dir / PEFT_WEIGHTS_NAME

    if isinstance(model_or_trainer, HFTrainer):
        model = model_or_trainer.model

        if (
                is_deepspeed_zero3_enabled() and
                is_deepspeed_module(model) and
                model_or_trainer.optimizer is not None
        ):
            # In DeepSpeed ZeRO3, huggingface Trainer saves full model checkpoint.
            # When using Fairscale, Deepspeed or PyTorch FSDP, optimizer is only initialized during Trainer.train;
            # to check if ZeRO3 is fully initialized, also need to check optimizer.
            model_or_trainer.save_checkpoint(str(checkpoint_dir))

        elif is_saving_process:
            # Need to manually save full checkpoint when not using DeepSpeed.
            save_model_helper(model, checkpoint_dir)

    elif isinstance(model_or_trainer, Module):
        model = model_or_trainer

        save_model_helper(model, checkpoint_dir)

    else:
        raise TypeError(f"\"{type(model_or_trainer)}\" is not a supported type.")

    barrier()

    # save PEFT model if do not exist
    if is_saving_process and isinstance(model, PeftModel) and not is_file(peft_checkpoint_path):
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")

        torch.save(
            get_peft_model_state_dict(model, state_dict=state_dict),
            str(peft_checkpoint_path)
        )

        del state_dict
        gc.collect()

    # all process should wait
    barrier()


def load_model_state_dict(checkpoint_dir: PathType) -> OrderedDict:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir / HF_WEIGHTS_NAME
    peft_checkpoint_path = checkpoint_dir / PEFT_WEIGHTS_NAME

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
            training_args: FinetuningArguments,
            test_dataset: Optional[Dataset] = None
    ):
        super().__init__(model, args)

        self.tokenizer = tokenizer
        self.training_args = training_args
        self.trainer = get_hf_trainer(
            args=self.args,
            model=self.model,
            tokenizer=self.tokenizer,
            training_args=self.training_args,
            eval_dataset=test_dataset
        )

        max_steps = self.trainer.args.max_steps
        num_train_epochs = self.trainer.args.num_train_epochs
        comm_round = int(self.args.comm_round)
        assert max_steps > 0 or num_train_epochs > 0, \
            f"at least 1 of max_steps and num_train_epochs should be positive, " \
            f"but got {max_steps} and {num_train_epochs}"
        assert max_steps <= 0 or max_steps >= comm_round, \
            f"max_steps = {max_steps} > 0 must be greater than comm_round = {comm_round}"

        if max_steps > 0:
            assert max_steps >= comm_round, f"required max_steps >= comm_round, but got {max_steps} < {comm_round}"
            step_threshold = int(math.ceil(max_steps / comm_round))
            epoch_threshold = math.inf
            self.log(f"step_threshold = {step_threshold}")
        elif num_train_epochs > 0:
            # TODO: verify
            step_threshold = math.inf
            epoch_threshold = num_train_epochs / comm_round
            self.log(f"epoch_threshold = {epoch_threshold}")
        else:
            raise ValueError(
                f"at least one of the `max_steps` and `num_train_epochs` should be positive, "
                f"but got {max_steps} and {num_train_epochs}"
            )

        self.trainer.add_callback(PauseResumeCallback(
            step_threshold=step_threshold,
            epoch_threshold=epoch_threshold
        ))

        barrier()
        self.log("initialized")

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.trainer.args.output_dir)

    def is_main_process(self) -> bool:
        return is_main_process(self.trainer)

    def log(self, message: str, stack_level: int = 1) -> None:
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

        peft_state_dict = load_model_state_dict(self.checkpoint_dir)

        self.log("finished")
        return OrderedDict(peft_state_dict)

    def set_model_params(self, model_parameters) -> None:
        self.log("start")

        model_parameters = to_device(model_parameters, device="cpu")

        barrier()
        set_peft_model_state_dict(self.model, model_parameters)
        barrier()

        self.log("finished")

    def on_before_local_training(self, train_data, device, args: Arguments) -> None:
        self.log("start")

        outputs = super().on_before_local_training(train_data, device, args)

        # update round_idx
        if hasattr(args, "round_idx"):
            self.round_idx = args.round_idx

        self.trainer.train_dataset = train_data

        if self.round_idx > 0:
            # TODO: remove once FedML integrated the change
            self.test(self.trainer.eval_dataset, device, args)

        self.log("finished")
        return outputs

    def train(self, train_data, device, args: Arguments) -> None:
        self.log("start")

        self.trainer.train()

        self.log("finished")

    def on_after_local_training(self, train_data, device, args: Arguments) -> None:
        self.log("start")

        outputs = super().on_after_local_training(train_data, device, args)

        self.log(f"saving model to \"{self.checkpoint_dir}\"")
        save_model_state_dict(self.trainer)

        self.log("finished")
        return outputs

    def test(self, test_data, device, args) -> None:
        self.log("start")

        if not self.is_run_test:
            self.log("skipped")
            return

        metrics = self.trainer.evaluate(eval_dataset=test_data, metric_key_prefix=f"client{self.args.rank}_eval")
        if self.is_main_process():
            mlops.log({**metrics, "round_idx": self.round_idx})

        self.log("finished")

    @property
    def is_run_test(self) -> bool:
        return getattr(self.args, "is_client_test", False)

    @property
    def round_idx(self) -> int:
        return getattr(self.args, "round_idx", -1)

    @round_idx.setter
    def round_idx(self, round_idx: int) -> None:
        setattr(self.args, "round_idx", round_idx)

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
            training_args: FinetuningArguments
    ):
        super().__init__(model, args)

        self.tokenizer = tokenizer
        self.training_args = training_args
        self.trainer = get_hf_trainer(
            args=self.args,
            model=self.model,
            tokenizer=self.tokenizer,
            training_args=self.training_args,
            # save peft adapted model weights
            callbacks=[SavePeftModelCallback]
        )

        # save config
        if should_process_save(self.trainer):
            # save model config before training
            save_config(model, self.checkpoint_dir)

        barrier()
        self.log("initialized")

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.trainer.args.output_dir)

    def is_main_process(self) -> bool:
        return is_main_process(self.trainer)

    def log(self, message: str, stack_level: int = 1) -> None:
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

        peft_state_dict = load_model_state_dict(self.checkpoint_dir)

        self.log("finished")
        return OrderedDict(peft_state_dict)

    def set_model_params(self, model_parameters) -> None:
        self.log("start")

        model_parameters = to_device(model_parameters, device="cpu")

        barrier()
        set_peft_model_state_dict(self.model, model_parameters)
        barrier()

        self.log("finished")

    def test(self, test_data, device, args: Arguments) -> None:
        self.log("start")

        if not self.is_run_test:
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
    def is_run_test(self) -> bool:
        return getattr(self.args, "is_aggregator_test", True)

    @property
    def round_idx(self) -> int:
        return getattr(self.args, "round_idx", -1)

    @round_idx.setter
    def round_idx(self, round_idx: int) -> None:
        setattr(self.args, "round_idx", round_idx)


def transform_data_to_fedml_format(args: Arguments, train_dataset: Dataset, test_dataset: Dataset):
    # TODO: scrutinize
    train_data_num = 0
    test_data_num = 0
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
        train_data_local_num_dict[args.rank - 1] = len(train_dataset)
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
        2
    )


def main(args: Arguments) -> None:
    # init device
    device = fedml.device.get_device(args)

    model_args, dataset_args = parse_hf_args((ModelArguments, DatasetArguments), args)

    tokenizer = get_tokenizer(
        model_args,
        add_special_tokens=getattr(args, "task", "finetune") == "instruction"
    )

    if args.local_rank == 0:
        # Initialize model before initializing TrainingArgs to load the full model in memory
        # This is required when using DeepSpeed Zero3 w/ FedLLM
        model = get_model(
            model_args,
            tokenizer_length=len(tokenizer),
            use_cache=not getattr(args, "gradient_checkpointing", False)
        )

        # save initial model. This is required for DeepSpeed
        save_model_state_dict(
            model_or_trainer=model,
            checkpoint_dir=args.output_dir,
            is_saving_process=True
        )
        del model
        gc.collect()
    barrier()

    training_args, *_ = parse_hf_args(FinetuningArguments, args)

    # update cross-silo hierarchical related settings
    if getattr(args, "use_customized_hierarchical", False):
        setattr(args, "proc_rank_in_silo", training_args.process_index)
        setattr(args, "rank_in_node", training_args.local_process_index)
        setattr(args, "process_id", training_args.process_index)

    model = get_model(
        model_args,
        tokenizer_length=len(tokenizer),
        use_cache=not getattr(args, "gradient_checkpointing", False)
    )

    if dataset_args.max_seq_length is None:
        dataset_args.max_seq_length = get_max_seq_length(model)
        setattr(args, "max_seq_length", dataset_args.max_seq_length)

    with training_args.main_process_first():
        train_dataset, test_dataset = get_dataset(
            dataset_args=dataset_args,
            tokenizer=tokenizer,
            seed=args.seed
        )

    # load data
    dataset = transform_data_to_fedml_format(args, train_dataset, test_dataset)

    # FedML trainer
    trainer = aggregator = None
    if args.role == "client":
        trainer = LLMTrainer(
            model=model,
            args=args,
            tokenizer=tokenizer,
            training_args=training_args,
            test_dataset=test_dataset
        )
    elif args.role == "server":
        aggregator = LLMAggregator(
            model=model,
            args=args,
            tokenizer=tokenizer,
            training_args=training_args
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
