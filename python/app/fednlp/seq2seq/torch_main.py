import logging

from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
)

import fedml
from data.data_loader import load
from fedml import FedMLRunner
from fedml.model.nlp.model_args import *
from trainer.seq2seq_trainer import MyModelTrainer as MySSTrainer
from trainer.seq2seq_aggregator import Seq2SeqAggregator

def create_model(args, device, output_dim=1):
    model_name = args.model
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (model_name, output_dim)
    )
    MODEL_CLASSES = {
        "seq2seq": {"bart": (BartConfig, BartForConditionalGeneration, BartTokenizer),}
    }
    try:
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.formulation][
            args.model_type
        ]
    except KeyError:
        raise Exception("such model or formulation does not exist currently!")
    model_args = Seq2SeqArgs()
    model_args.model_name = args.model
    model_args.model_type = args.model_type
    # model_args.load(model_args.model_name)
    # model_args.num_labels = num_labels
    model_args.update_from_dict(
        {
            "fl_algorithm": args.federated_optimizer,
            "freeze_layers": args.freeze_layers,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "do_lower_case": args.do_lower_case,
            "manual_seed": args.random_seed,
            # for ignoring the cache features.
            "reprocess_input_data": args.reprocess_input_data,
            "overwrite_output_dir": True,
            "max_seq_length": args.max_seq_length,
            "train_batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "evaluate_during_training": False,  # Disabled for FedAvg.
            "evaluate_during_training_steps": args.evaluate_during_training_steps,
            "fp16": args.fp16,
            "data_file_path": args.data_file_path,
            "partition_file_path": args.partition_file_path,
            "partition_method": args.partition_method,
            "dataset": args.dataset,
            "output_dir": args.output_dir,
            "is_debug_mode": args.is_debug_mode,
            "fedprox_mu": args.fedprox_mu,
            "optimizer": args.client_optimizer,
        }
    )

    # model_args.config["num_labels"] = num_labels
    tokenizer = [None, None]
    tokenizer[0] = tokenizer_class.from_pretrained(args.model)
    tokenizer[1] = tokenizer[0]
    # model_args["num_labels"] = output_dim
    model_config = {}
    config = config_class.from_pretrained(args.model, **model_config)
    model = model_class.from_pretrained(args.model, config=config)
    print("reached_here")
    trainer = MySSTrainer(model_args, device, model, tokenizer=tokenizer)
    return model, trainer, model_args


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load(args)
    # args.num_labels = output_dim
    # load model and trainer
    model, trainer, model_args = create_model(args, output_dim)
    aggregator = Seq2SeqAggregator(model, model_args)
    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()
