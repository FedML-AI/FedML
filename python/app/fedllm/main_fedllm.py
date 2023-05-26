from collections import OrderedDict
import logging
import math
from pathlib import Path

import fedml
from fedml import FedMLRunner, mlops
from fedml.arguments import Arguments
from fedml.core import ClientTrainer, ServerAggregator
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import torch.cuda
from transformers import HfArgumentParser, Trainer as HfTrainer, TrainingArguments

from src.constants import DEFAULT_MAX_SEQ_LENGTH
from src.trainer_callback import PauseResumeCallback
from src.utils import (
    process_state_dict,
    save_config,
    should_process_save,
    to_device,
)
from train import (
    DataArguments,
    get_data_collator,
    get_dataset,
    get_model,
    get_max_seq_length,
    get_tokenizer,
    ModelArguments,
    ModelType,
    SavePeftModelCallback,
    TokenizerType,
)


def _parse_args(args: Arguments) -> Arguments:
    if args.role == "client":
        if hasattr(args, "client_dataset_path"):
            args.dataset_path = args.client_dataset_path
        # disable logging for client
        setattr(args, "report_to", "none")
        setattr(args, "disable_tqdm", True)

    if isinstance(args.dataset_path, (tuple, list)):
        args.dataset_path = [
            p.format(rank=args.rank, client_num_in_total=args.client_num_in_total)
            for p in args.dataset_path
        ]

    if torch.cuda.device_count() == 0:
        logging.warning(f"{args.role} rank {args.rank} does not have GPU! Fallback to CPU mode.")
        setattr(args, "deepspeed", None)

    return args


def get_hf_trainer(args: Arguments, model: ModelType, tokenizer: TokenizerType, **kwargs) -> HfTrainer:
    args_dict = dict(args.__dict__)
    # TODO: scrutinize
    if not args.using_gpu or torch.cuda.device_count() == 1:
        args_dict.pop("local_rank", None)
        args_dict.pop("device", None)
    training_args, *_ = HfArgumentParser(TrainingArguments).parse_dict(args_dict, allow_extra_keys=True)

    return HfTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=get_data_collator(tokenizer, getattr(args, "max_seq_length", DEFAULT_MAX_SEQ_LENGTH)),
        **kwargs
    )


def get_model_state_dict(trainer: HfTrainer, checkpoint_dir: Path) -> OrderedDict:
    with trainer.args.main_process_first():
        checkpoint_path = checkpoint_dir / "pytorch_model.bin"
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    return checkpoint


class LLMTrainer(ClientTrainer):
    def __init__(
            self,
            model: ModelType,
            args: Arguments,
            tokenizer: TokenizerType,
            model_args: ModelArguments
    ):
        super().__init__(model, args)

        self.tokenizer = tokenizer
        self.model_args = model_args
        self.trainer = get_hf_trainer(self.args, self.model, self.tokenizer)

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
            logging.info(f"step_threshold = {step_threshold}")
        elif num_train_epochs > 0:
            # TODO: verify
            step_threshold = math.inf
            epoch_threshold = num_train_epochs / comm_round
            logging.info(f"epoch_threshold = {epoch_threshold}")
        else:
            raise ValueError(
                f"at least one of the `max_steps` and `num_train_epochs` should be positive, "
                f"but got {max_steps} and {num_train_epochs}"
            )

        self.trainer.add_callback(PauseResumeCallback(
            step_threshold=step_threshold,
            epoch_threshold=epoch_threshold
        ))

        self.temp_ckpt_dir = Path(self.trainer.args.output_dir) / f"node{self.args.rank}_tmp"
        # this is required for DeepSpeed
        self.trainer.save_model(str(self.temp_ckpt_dir))

    def get_model_params(self) -> OrderedDict:
        state_dict = get_model_state_dict(self.trainer, self.temp_ckpt_dir)
        peft_state_dict = to_device(get_peft_model_state_dict(self.model, state_dict=state_dict), device="cpu")
        return OrderedDict(peft_state_dict)

    def set_model_params(self, model_parameters) -> None:
        model_parameters = to_device(model_parameters, device="cpu")
        model_parameters = process_state_dict(model_parameters, get_peft_model_state_dict(self.model))

        set_peft_model_state_dict(self.model, model_parameters)

    def on_before_local_training(self, train_data, device, args: Arguments) -> None:
        super().on_before_local_training(train_data, device, args)

        # TODO: replace args?
        # update round_idx
        if hasattr(args, "round_idx"):
            setattr(self.args, "round_idx", args.round_idx)

        self.trainer.train_dataset = train_data

        if getattr(self.args, "round_idx", -1) > 0:
            # TODO: verify model, model_wrapped, deepspeed, optimizer, lr_scheduler after reset
            # turn off TrainingArguments.deepspeed to avoid duplicated initializations
            self.trainer.args.deepspeed = None

    def train(self, train_data, device, args: Arguments) -> None:
        self.trainer.train()

    def on_after_local_training(self, train_data, device, args: Arguments) -> None:
        super().on_after_local_training(train_data, device, args)
        self.trainer.save_model(str(self.temp_ckpt_dir))
        # recover TrainingArguments.deepspeed
        self.trainer.args.deepspeed = self.args.deepspeed


class LLMAggregator(ServerAggregator):
    def __init__(
            self,
            model: ModelType,
            args: Arguments,
            tokenizer: TokenizerType
    ):
        super().__init__(model, args)

        self.tokenizer = tokenizer
        self.model = model

        self.trainer = get_hf_trainer(
            args=self.args,
            model=self.model,
            tokenizer=self.tokenizer,
            # save peft adapted model weights
            callbacks=[SavePeftModelCallback]
        )
        self.temp_ckpt_dir = Path(self.trainer.args.output_dir) / f"node{self.args.rank}_tmp"

        # save config
        if should_process_save(self.trainer):
            # save model config before training
            save_config(model, self.trainer.args.output_dir)

    def get_model_params(self) -> OrderedDict:
        state_dict = get_model_state_dict(self.trainer, self.temp_ckpt_dir)
        peft_state_dict = to_device(get_peft_model_state_dict(self.model, state_dict=state_dict), device="cpu")
        return OrderedDict(peft_state_dict)

    def set_model_params(self, model_parameters) -> None:
        # TODO: verify DeepSpeed support
        model_parameters = to_device(model_parameters, device="cpu")
        model_parameters = process_state_dict(model_parameters, get_peft_model_state_dict(self.model))

        set_peft_model_state_dict(self.model, model_parameters)

    def test(self, test_data, device, args: Arguments) -> None:
        # update epoch, global_step for logging
        self.trainer.state.epoch = self.args.round_idx
        self.trainer.state.global_step = self.args.round_idx
        metrics = self.trainer.evaluate(eval_dataset=test_data)
        mlops.log({**metrics, "round_idx": args.round_idx})


def transform_data_to_fedml_format(args: Arguments, dataset):
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
        test_data_global = dataset
    else:
        # client data
        train_data_local_num_dict[args.rank - 1] = len(dataset)
        train_data_local_dict[args.rank - 1] = dataset
        test_data_local_dict[args.rank - 1] = None  # we do not do test on the client
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

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, dataset_args = parser.parse_dict(dict(args.__dict__), allow_extra_keys=True)

    # TODO: init model here?
    tokenizer = get_tokenizer(model_args.model_name)
    model = get_model(
        model_args,
        tokenizer_length=len(tokenizer),
        use_cache=not getattr(args, "gradient_checkpointing", False)
    )

    if dataset_args.max_seq_length is None:
        dataset_args.max_seq_length = get_max_seq_length(model)
        setattr(args, "max_seq_length", dataset_args.max_seq_length)

    train_dataset, test_dataset = get_dataset(
        dataset_path=dataset_args.dataset_path,
        tokenizer=tokenizer,
        max_length=dataset_args.max_seq_length,
        seed=args.seed,
        test_dataset_size=dataset_args.test_dataset_size
    )

    # load data
    if args.rank == 0:
        dataset = test_dataset
        print(f"Test data size: {dataset.num_rows:,}")
    else:
        dataset = train_dataset
        print(f"Train data size: {dataset.num_rows:,}")
    dataset = transform_data_to_fedml_format(args, dataset)

    # FedML trainer
    trainer = LLMTrainer(model=model, args=args, tokenizer=tokenizer, model_args=model_args)
    aggregator = LLMAggregator(model=model, args=args, tokenizer=tokenizer)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()


if __name__ == "__main__":
    # init FedML framework
    main(args=_parse_args(fedml.init()))
