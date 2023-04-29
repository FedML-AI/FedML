from collections import OrderedDict

import fedml
from fedml import FedMLRunner
from fedml.arguments import Arguments
from fedml.core import ClientTrainer, ServerAggregator
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import torch.cuda
from transformers import HfArgumentParser, Trainer as HfTrainer, TrainingArguments

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


def get_hf_trainer(args: Arguments, model: ModelType, tokenizer: TokenizerType, **kwargs) -> HfTrainer:
    args_dict = dict(args.__dict__)
    # TODO: scrutinize
    if not args.using_gpu or torch.cuda.device_count() == 1:
        args_dict.pop("local_rank", None)
    training_args, *_ = HfArgumentParser(TrainingArguments).parse_dict(args_dict, allow_extra_keys=True)

    return HfTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=get_data_collator(tokenizer),
        **kwargs
    )


class LLMTrainer(ClientTrainer):
    def __init__(
            self,
            model: ModelType,
            args: Arguments,
            tokenizer: TokenizerType
    ):
        super().__init__(model, args)

        self.tokenizer = tokenizer
        self.model = model

    def get_model_params(self) -> OrderedDict:
        return OrderedDict(get_peft_model_state_dict(self.model))

    def set_model_params(self, model_parameters) -> None:
        set_peft_model_state_dict(self.model, model_parameters)

    def train(self, train_data, device, args: Arguments) -> None:
        trainer = get_hf_trainer(args, self.model, self.tokenizer, train_dataset=train_data)
        trainer.train()


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
            args=args,
            model=self.model,
            tokenizer=self.tokenizer,
            # save peft adapted model weights
            callbacks=[SavePeftModelCallback]
        )

    def get_model_params(self) -> OrderedDict:
        return OrderedDict(get_peft_model_state_dict(self.model))

    def set_model_params(self, model_parameters) -> None:
        set_peft_model_state_dict(self.model, model_parameters)

    def test(self, test_data, device, args: Arguments) -> None:
        self.trainer.evaluate(eval_dataset=test_data)


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
    trainer = LLMTrainer(model=model, args=args, tokenizer=tokenizer)
    aggregator = LLMAggregator(model=model, args=args, tokenizer=tokenizer)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()


if __name__ == "__main__":
    # init FedML framework
    main(args=fedml.init())
