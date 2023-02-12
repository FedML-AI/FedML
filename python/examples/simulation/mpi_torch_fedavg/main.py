import argparse
import logging
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))


import fedml
from fedml import FedMLRunner
from fedml.arguments import Arguments
from fedml.model.cv.resnet import resnet20
from fedml.model.cv.resnet_cifar import resnet18_cifar
from fedml.model.cv.resnet_gn import resnet18
from fedml.model.cv.resnet_torch import resnet50
from fedml.model.cv.resnet_torch import resnet18 as resnet18_torch


from fedml.arguments import Arguments

from fedml import FedMLRunner


import fedml.simulation.mpi.fedavg as fedavg
# print(f"fedavg.__file__: {fedavg.__file__}")

import fedml.simulation.mpi.mpi
from fedml.simulation.mpi.mpi.DistributedAPI import FedML_distributed
from fedml.simulation.mpi.mpi_seq.DistributedAPI import FedML_distributed_seq

from model_trainer import ModelTrainerCLS
from aggregator import DefaultServerAggregator


def str2bool(v):
    if isinstance(v, bool):
        return v
    # if v.lower() in ('yes', 'true', 't', 'y', '1'):
    if isinstance(v, str) and v.lower() in ('true', 'True'):
        return True
    elif isinstance(v, str) and v.lower() in ('false', 'False'):
        return False
    else:
        return v
        # raise argparse.ArgumentTypeError('Boolean value expected.')



def add_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file", "--cf", help="yaml configuration file", type=str, default="",
    )

    # default arguments
    parser.add_argument("--run_id", type=str, default="0")

    # default arguments
    parser.add_argument("--rank", type=int, default=0)

    # default arguments
    parser.add_argument("--local_rank", type=int, default=0)

    # For hierarchical scenario
    parser.add_argument("--node_rank", type=int, default=0)

    # default arguments
    parser.add_argument("--role", type=str, default="client")

    # cluster arguments
    parser.add_argument("--worker_num", type=int, default=8)
    parser.add_argument("--gpu_util_parse", type=str, default="localhost:2,1,1,1,1,1,1,1")

    # Model and dataset
    parser.add_argument("--model", type=str, default="resnet18_cifar")
    parser.add_argument("--group_norm_channels", type=int, default=32)
    parser.add_argument("--dataset", type=str, default="fed_cifar100")
    parser.add_argument(
        "--data_cache_dir", type=str, default="/home/chaoyanghe/zhtang_FedML/python/fedml/data/fed_cifar100/datasets"
    )
    parser.add_argument(
        "--net_dataidx_map_file", type=str, default=None
    )

    # Training arguments
    parser.add_argument("--federated_optimizer", type=str, default="FedAvg_seq")
    parser.add_argument("--client_num_in_total", type=int, default=500)
    parser.add_argument("--client_num_per_round", type=int, default=10)
    parser.add_argument("--comm_round", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--client_optimizer", type=str, default="sgd")
    parser.add_argument("--learning_rate", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--server_optimizer", type=str, default="sgd")
    parser.add_argument("--server_lr", type=float, default=1.0)
    parser.add_argument("--server_momentum", type=float, default=0.9)
    # args, unknown = parser.parse_known_args()

    parser.add_argument("--mimelite", type=str, default="False")

    parser.add_argument("--frequency_of_the_test", type=int, default=10)

    #
    parser.add_argument("--feddyn_alpha", type=float, default=0.01)

    parser.add_argument("--feddlc_download_dense", type=str, default="False")

    parser.add_argument("--compression", type=str, default="no")
    parser.add_argument("--compression_sigma_scale", type=int, default=3)
    parser.add_argument("--compression_sparse_ratio", type=float, default=0.01)
    parser.add_argument("--compression_quantize_level", type=int, default=8)
    parser.add_argument("--compression_is_biased", type=str, default="False")

    parser.add_argument("--down_compression", type=str, default="no")
    parser.add_argument("--down_compression_sigma_scale", type=int, default=3)
    parser.add_argument("--down_compression_sparse_ratio", type=float, default=0.1)
    parser.add_argument("--down_compression_quantize_level", type=int, default=8)
    parser.add_argument("--down_compression_is_biased", type=str, default="False")

    parser.add_argument("--compression_warmup_round", type=int, default=20)

    parser.add_argument("--aggregate_seq", type=str, default="False")
    parser.add_argument("--hierarchical_agg", type=str, default="False")


    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--override_cmd_args", action="store_true")
    parser.add_argument("--tag", type=str, default="debug")
    parser.add_argument("--exp_name", type=str, default="ready")
    parser.add_argument("--device_tag", type=str, default="lambda3")

    parser.add_argument("--enable_wandb", type=str, default="True")
    parser.add_argument("--run_name", type=str, default="fl-compression")


    parser.add_argument("--local_cache", type=str, default="False")
    parser.add_argument("--simulation_schedule", type=str, default=None)
    parser.add_argument("--runtime_est_mode", type=str, default="history")

    parser.add_argument("--simulation_gpu_hetero", type=str, default=None)
    parser.add_argument("--gpu_hetero_ratio", type=float, default=1.0)
    parser.add_argument("--simulation_environment_hetero", type=str, default=None)
    parser.add_argument("--environment_hetero_ratio", type=float, default=1.0)

    parser.add_argument("--tracking_runtime", type=str, default="False")

    args = parser.parse_args()

    for key in args.__dict__.keys():
        args.__dict__[key] = str2bool(args.__dict__[key])
    return args


def load_arguments(training_type=None, comm_backend=None):
    cmd_args = add_args()
    logging.info(f"cmd_args: {cmd_args}")

    # Load all arguments from YAML config file
    args = Arguments(cmd_args, training_type, comm_backend, override_cmd_args=cmd_args.override_cmd_args)

    # if not hasattr(args, "worker_num"):
    #     args.worker_num = args.client_num_per_round

    # os.path.expanduser() method in Python is used
    # to expand an initial path component ~( tilde symbol)
    # or ~user in the given path to user’s home directory.
    if hasattr(args, "data_cache_dir"):
        args.data_cache_dir = os.path.expanduser(args.data_cache_dir)
    if hasattr(args, "data_file_path"):
        args.data_file_path = os.path.expanduser(args.data_file_path)
    if hasattr(args, "partition_file_path"):
        args.partition_file_path = os.path.expanduser(args.partition_file_path)
    if hasattr(args, "part_file"):
        args.part_file = os.path.expanduser(args.part_file)

    args.rank = int(args.rank)
    return args


if __name__ == "__main__":
    # init FedML framework
    args = load_arguments(fedml._global_training_type, fedml._global_comm_backend)
    logging.info(f"args: {args}")
    args = fedml.init(args)
    logging.info(f"args: {args}")
    # args = fedml.init()
    logging.info(f"args.process_id: {args.process_id}, args.worker_num: {args.worker_num}")

    # init device
    device = fedml.device.get_device(args)
    logging.info(
        f"======================================================== \
        process_id: {args.process_id}, device: {device} =============="
    )

    # load data
    if args.dataset == "reddit":
        from fedml.data.reddit.data_loader import load_partition_data_reddit
        dataset = load_partition_data_reddit(
            args,
            args.dataset,
            args.data_cache_dir,
            args.partition_method,
            args.partition_alpha,
            args.client_num_in_total,
            args.batch_size,
            n_proc_in_silo=0,
        )
        output_dim = 10000
    else:
        dataset, output_dim = fedml.data.load(args)

    if args.dataset == "femnist":
        in_channels = 1
    else:
        in_channels = 3

    # load model (the size of MNIST image is 28 x 28)
    if args.model == "resnet18":
        logging.info("ResNet18_GN")
        model = resnet18(group_norm=args.group_norm_channels, num_classes=output_dim)
    elif args.model == "resnet50":
        logging.info("ResNet50_GN")
        model = resnet50(num_classes=output_dim, in_channels=in_channels, group_norm=args.group_norm_channels)
        # model = resnet50(group_norm=args.group_norm_channels, num_classes=output_dim)
    elif args.model == "resnet20":
        model = resnet20(class_num=output_dim)
    elif args.model == "resnet18_torch":
        model = resnet18_torch(num_classes=output_dim, in_channels=in_channels)
    elif args.model == "resnet18_cifar":
        logging.info("resnet18_cifar")
        model = resnet18_cifar(group_norm=args.group_norm_channels, num_classes=output_dim)
    elif args.model == "albert-base-v2":
        from transformers import (AdamW, AlbertTokenizer, AutoConfig,
                                AutoModelWithLMHead, AutoTokenizer,
                                MobileBertForPreTraining)
        config = AutoConfig.from_pretrained(
            os.path.join(args.data_cache_dir, args.model+'-config.json'))
        model = AutoModelWithLMHead.from_config(config)
    else:
        model = fedml.model.create(args, output_dim)


    # if args.dataset == "reddit":
    #     from reddit_trainer import RedditTrainer
    #     from reddit_aggregator import RedditAggregator
    #     client_trainer = RedditTrainer(model, args)
    #     server_aggregator = RedditAggregator(model, args)
    #     fedml_runner = FedMLRunner(args, device, dataset, model,
    #         client_trainer=client_trainer, server_aggregator=server_aggregator)
    #     fedml_runner.run()
    # else:
    #     # start training
    #     fedml_runner = FedMLRunner(args, device, dataset, model)
    #     fedml_runner.run()
    #     # simulator = SimulatorMPI(args, device, dataset, model)
    #     # simulator.run()


    [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ] = dataset
    client_trainer = ModelTrainerCLS(model, args)
    logging.info(f"Server: 1, Workers: {args.worker_num-1}")
    server_aggregator = DefaultServerAggregator(
        train_data_global,
        test_data_global,
        train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        args.worker_num - 1,
        device,
        args,
        model)

    fedml_runner = FedMLRunner(args, device, dataset, model,
                client_trainer=client_trainer, server_aggregator=server_aggregator)
    fedml_runner.run()
    # simulator = SimulatorMPI(args, device, dataset, model)
    # simulator.run()



    # if hasattr(args, "hierarchical_agg") and args.hierarchical_agg:
    #     FedML_distributed_seq(args, args.process_id, args.worker_num, args.comm,
    #                     device, dataset, model)
    # else:
    #     FedML_distributed(args, args.process_id, args.worker_num, args.comm,
    #                     device, dataset, model)





