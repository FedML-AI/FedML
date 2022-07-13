import os

import numpy as np
import torch
from transformers import (
    BertConfig,
    BertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
)
from data.model_args import *
from fedml.data.fednlp.base.data_manager.base_data_manager import BaseDataManager
from data.seq_tagging_data_manager import (
    SequenceTaggingDataManager,
)

from data.seq_tagging_preprocessor import (
    TLMPreprocessor as STPreprocessor,
)

# from fedml.data.FederatedEMNIST.data_loader import load_partition_data_federated_emnist
# from fedml.data.ImageNet.data_loader import load_partition_data_ImageNet
# from fedml.data.Landmarks.data_loader import load_partition_data_landmarks
# from fedml.data.MNIST.data_loader import load_partition_data_mnist
# from fedml.data.cifar10.data_loader import load_partition_data_cifar10
# from fedml.data.cifar100.data_loader import load_partition_data_cifar100
# from fedml.data.cinic10.data_loader import load_partition_data_cinic10
# from fedml.data.fed_cifar100.data_loader import load_partition_data_federated_cifar100
# from fedml.data.fed_shakespeare.data_loader import (
#     load_partition_data_federated_shakespeare,
# )
# from fedml.data.shakespeare.data_loader import load_partition_data_shakespeare
# from fedml.data.stackoverflow_lr.data_loader import (
#     load_partition_data_federated_stackoverflow_lr,
# )
# from fedml.data.stackoverflow_nwp.data_loader import (
#     load_partition_data_federated_stackoverflow_nwp,
# )
#
# from .MNIST.data_loader import download_mnist
# from .edge_case_examples.data_loader import load_poisoned_dataset
import logging


def load(args):
    return load_synthetic_data(args)


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


def load_synthetic_data(args):
    dataset_name = args.dataset
    # check if the centralized training is enabled
    centralized = (
        True
        if (args.client_num_in_total == 1 and args.training_type != "cross_silo")
        else False
    )

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False
    logging.info("load_data. dataset_name = %s" % dataset_name)
    attributes = BaseDataManager.load_attributes(args.data_file_path)
    num_labels = len(attributes["label_vocab"])
    class_num = num_labels
    model_args = SeqTaggingArgs()
    model_args.model_name = args.model
    model_args.model_type = args.model_type
    # model_args.load(model_args.model_name)
    model_args.num_labels = num_labels
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
        }
    )

    # model_args.config["num_labels"] = num_labels
    if args.model_type == "bert":
        tokenizer_class = BertTokenizer
    elif args.model_type == "distilbert":
        tokenizer_class = DistilBertTokenizer
    tokenizer = tokenizer_class.from_pretrained(
        args.model, do_lower_case=args.do_lower_case
    )
    preprocessor = STPreprocessor(
        args=model_args, label_vocab=attributes["label_vocab"], tokenizer=tokenizer
    )
    dm = SequenceTaggingDataManager(
        args, model_args, preprocessor, 0, args.client_num_per_round
    )

    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        num_clients,
    ) = dm.load_federated_data()

    if centralized:
        train_data_local_num_dict = {
            0: sum(
                user_train_data_num
                for user_train_data_num in train_data_local_num_dict.values()
            )
        }
        train_data_local_dict = {
            0: [
                batch
                for cid in sorted(train_data_local_dict.keys())
                for batch in train_data_local_dict[cid]
            ]
        }
        test_data_local_dict = {
            0: [
                batch
                for cid in sorted(test_data_local_dict.keys())
                for batch in test_data_local_dict[cid]
            ]
        }
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {
            cid: combine_batches(train_data_local_dict[cid])
            for cid in train_data_local_dict.keys()
        }
        test_data_local_dict = {
            cid: combine_batches(test_data_local_dict[cid])
            for cid in test_data_local_dict.keys()
        }
        args.batch_size = args_batch_size

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num


def load_poisoned_dataset_from_edge_case_examples(args):
    return load_poisoned_dataset(args=args)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=0)

    parser.add_argument("--is_debug_mode", default=0, type=int, help="is_debug_mode")

    # Data related
    # TODO: list all dataset names:
    parser.add_argument(
        "--dataset",
        type=str,
        default="20news",
        metavar="N",
        help="dataset used for training",
    )

    parser.add_argument(
        "--data_file_path",
        type=str,
        default="/home/dbc/fednlp_data/data_files/20news_data.h5",
        help="data h5 file path",
    )

    parser.add_argument(
        "--partition_file_path",
        type=str,
        default="/home/dbc/fednlp_data/partition_files/20news_partition.h5",
        help="partition h5 file path",
    )

    parser.add_argument(
        "--partition_method", type=str, default="uniform", help="partition method"
    )

    # Model related
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        metavar="N",
        help="transformer model type",
    )

    parser.add_argument(
        "--model_class",
        type=str,
        default="transformer",
        metavar="N",
        help="model class",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        metavar="N",
        help="transformer model name",
    )
    parser.add_argument(
        "--do_lower_case",
        type=bool,
        default=True,
        metavar="N",
        help="transformer model name",
    )

    # Learning related
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        metavar="N",
        help="input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        metavar="N",
        help="input batch size for evaluation (default: 8)",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        metavar="N",
        help="maximum sequence length (default: 128)",
    )

    parser.add_argument(
        "--n_gpu", type=int, default=1, metavar="EP", help="how many gpus will be used "
    )

    parser.add_argument(
        "--fp16", default=False, action="store_true", help="if enable fp16 for training"
    )
    parser.add_argument(
        "--manual_seed", type=int, default=42, metavar="N", help="random seed"
    )

    # IO related
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/",
        metavar="N",
        help="path to save the trained results and ckpts",
    )

    # Federated Learning related
    parser.add_argument(
        "--federated_optimizer",
        type=str,
        default="FedAvg",
        help="Algorithm list: FedAvg; FedOPT; FedProx ",
    )

    parser.add_argument(
        "--backend", type=str, default="MPI", help="Backend for Server and Client"
    )

    parser.add_argument(
        "--comm_round",
        type=int,
        default=10,
        help="how many round of communications we shoud use",
    )

    parser.add_argument(
        "--is_mobile",
        type=int,
        default=1,
        help="whether the program is running on the FedML-Mobile server side",
    )

    parser.add_argument(
        "--client_num_in_total",
        type=int,
        default=-1,
        metavar="NN",
        help="number of clients in a distributed cluster",
    )

    parser.add_argument(
        "--client_num_per_round",
        type=int,
        default=4,
        metavar="NN",
        help="number of workers",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="EP",
        help="how many epochs will be trained locally",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        metavar="EP",
        help="how many steps for accumulate the loss.",
    )

    parser.add_argument(
        "--client_optimizer",
        type=str,
        default="adam",
        help="Optimizer used on the client. This field can be the name of any subclass of the torch Opimizer class.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate on the client (default: 0.001)",
    )

    parser.add_argument(
        "--weight_decay", type=float, default=0, metavar="N", help="L2 penalty"
    )

    parser.add_argument(
        "--clip_grad_norm", type=int, default=0, metavar="N", help="L2 penalty"
    )

    parser.add_argument(
        "--server_optimizer",
        type=str,
        default="sgd",
        help="Optimizer used on the server. This field can be the name of any subclass of the torch Opimizer class.",
    )

    parser.add_argument(
        "--server_lr",
        type=float,
        default=0.1,
        help="server learning rate (default: 0.001)",
    )

    parser.add_argument(
        "--server_momentum", type=float, default=0, help="server momentum (default: 0)"
    )

    parser.add_argument(
        "--fedprox_mu", type=float, default=1, help="server momentum (default: 1)"
    )

    parser.add_argument(
        "--evaluate_during_training",
        default=False,
        metavar="EP",
        help="the frequency of the evaluation during training",
    )

    parser.add_argument(
        "--evaluate_during_training_steps",
        type=int,
        default=100,
        metavar="EP",
        help="the frequency of the evaluation during training",
    )

    parser.add_argument(
        "--frequency_of_the_test",
        type=int,
        default=1,
        help="the frequency of the algorithms",
    )

    # GPU device management
    parser.add_argument(
        "--gpu_mapping_file",
        type=str,
        default="gpu_mapping.yaml",
        help="the gpu utilization file for servers and clients. If there is no \
                    gpu_util_file, gpu will not be used.",
    )

    parser.add_argument(
        "--gpu_mapping_key",
        type=str,
        default="mapping_default",
        help="the key in gpu utilization file",
    )

    parser.add_argument("--ci", type=int, default=0, help="CI")

    # cached related
    parser.add_argument(
        "--reprocess_input_data", action="store_true", help="whether generate features"
    )

    # freeze related
    parser.add_argument(
        "--freeze_layers", type=str, default="", metavar="N", help="freeze which layers"
    )
    args = parser.parse_args()
    dataset, class_num = load(args)
