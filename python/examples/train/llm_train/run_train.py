from datetime import timedelta
import logging
from pathlib import Path
from timeit import default_timer as timer

from fedml.train.llm.configurations import DatasetArguments, ExperimentArguments, ModelArguments
from fedml.train.llm.hf_trainer import HFTrainer
from fedml.train.llm.modeling_utils import get_data_collator
from fedml.train.llm.train_utils import (
    get_dataset,
    get_max_seq_length,
    get_model,
    get_tokenizer,
)
from fedml.train.llm.utils import parse_hf_args, save_config


def train() -> None:
    # configs
    model_args, dataset_args, training_args = parse_hf_args((ModelArguments, DatasetArguments, ExperimentArguments))
    training_args.add_and_verify_args(model_args, dataset_args)

    # prepare models
    logging.info(f"Loading tokenizer for \"{model_args.model_name_or_path}\"")
    tokenizer = get_tokenizer(model_args)

    logging.info(f"Loading model for \"{model_args.model_name_or_path}\"")
    model = get_model(model_args, tokenizer_length=len(tokenizer), use_cache=not training_args.gradient_checkpointing)

    if dataset_args.max_seq_length is None:
        dataset_args.max_seq_length = get_max_seq_length(model)

    # dataset
    with training_args.main_process_first(local=True):
        train_dataset, test_dataset, eval_dataset = get_dataset(
            dataset_args=dataset_args,
            tokenizer=tokenizer,
            seed=training_args.seed,
            is_local_main_process=training_args.local_process_index == 0
        )

    trainer = HFTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=get_data_collator(
            tokenizer,
            response_template=dataset_args.response_template
        )
    )

    # log training time
    start_time = timer()

    if training_args.do_train:
        final_output_dir = Path(training_args.output_dir) / "final"

        if trainer.args.should_save:
            # save model config before training
            save_config(model, final_output_dir)

        logging.info("Training")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

        logging.info(f"Saving model to \"{final_output_dir}\"")
        trainer.save_checkpoint(final_output_dir)

    # log training time
    end_time = timer()
    logging.info(f"[{training_args.process_index}] total training time: {timedelta(seconds=end_time - start_time)}")

    if training_args.do_predict:
        logging.info("Evaluating")
        logging.info(trainer.evaluate(test_dataset))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    train()
