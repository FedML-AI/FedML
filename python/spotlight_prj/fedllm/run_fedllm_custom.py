#!/usr/bin/env python3
"""Custom FedLLM runner that properly handles non-PEFT models.

This runner uses FullModelLLMTrainer and FullModelLLMAggregator instead of the default
classes to fix the issue with peft_type="none" configuration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import fedml
from run_fedllm import main, _parse_args, transform_data_to_fedml_format, save_checkpoint
from custom_trainer import FullModelLLMTrainer, FullModelLLMAggregator

# Import everything else we need
from typing import Any, Dict, Optional
from torch.nn import Module
from fedml import FedMLRunner
from fedml.train.llm.configurations import DatasetArguments, ExperimentArguments, ModelArguments
from fedml.train.llm.distributed import barrier
from fedml.train.llm.train_utils import (
    get_dataset,
    get_model,
    get_max_seq_length,
    get_tokenizer,
)
from fedml.train.llm.utils import parse_hf_args, save_config
from fedml.train.llm.typing import PathType
import gc
from pathlib import Path
import torch
from peft import PeftModel
from transformers import PreTrainedModel
from transformers.utils import WEIGHTS_NAME as HF_WEIGHTS_NAME


def _save_checkpoint(
        model: Module,
        checkpoint_dir: PathType,
        state_dict: Optional[Dict[str, Any]] = None
) -> None:
    """Custom checkpoint saving that ensures pytorch_model.bin is created."""
    if state_dict is None:
        state_dict = model.state_dict()

    # Always produce a single-file checkpoint so that downstream loading logic
    # can simply look for ``adapter_model.bin`` (PEFT) or ``pytorch_model.bin``
    # without worrying about Hugging Face sharding.

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, PeftModel):
        filename = "adapter_model.bin"
    else:
        filename = HF_WEIGHTS_NAME  # "pytorch_model.bin"

    torch.save(state_dict, str(checkpoint_dir / filename))


# Monkey patch the _save_checkpoint function in the imported module
import run_fedllm
run_fedllm._save_checkpoint = _save_checkpoint


def main_custom(args):
    """Main function with custom trainer/aggregator for full model training."""
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

    # FedML trainer - USE CUSTOM CLASSES HERE
    trainer = aggregator = None
    if args.role == "client":
        trainer = FullModelLLMTrainer(  # Use custom trainer
            model=model,
            args=args,
            tokenizer=tokenizer,
            training_args=training_args,
            model_args=model_args,
            dataset_args=dataset_args
        )
    elif args.role == "server":
        aggregator = FullModelLLMAggregator(  # Use custom aggregator
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
    main_custom(args=_parse_args(fedml.init())) 