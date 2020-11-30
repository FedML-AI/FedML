import argparse
import logging
import os
import random
import socket
import sys
import datetime

import numpy as np
import psutil
import setproctitle
import torch
import wandb

# add the FedML root directory to the python path

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.coco.data_loader import load_partition_data_distributed_coco, load_partition_data_coco
from fedml_api.data_preprocessing.pascal_voc.data_loader import load_partition_data_distributed_pascal_voc, \
    load_partition_data_pascal_voc
from fedml_api.model.cv.deeplabV3 import DeeplabTransformer
from fedml_api.distributed.fedseg.FedSegAPI import FedML_init, FedML_FedSeg_distributed
from fedml_api.distributed.fedseg.utils import count_parameters

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='deeplab_transformer', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--backbone', type=str, default='resnet',
                        help='employ with backbone (default: xception)')

    parser.add_argument('--backbone_pretrained', type=bool, default=True,
                        help='pretrained backbone (default: True)')

    parser.add_argument('--backbone_freezed', type=bool, default=False,
                        help='Freeze backbone to extract features only once (default: False)')

    parser.add_argument('--extract_test', type=bool, default=False,
                        help='Extract Feature Maps of test data (default: False)')

    parser.add_argument('--outstride', type=int, default=8,
                        help='network output stride (default: 16)')

    # # TODO: Remove this argument
    # parser.add_argument('--categories', type=str, default='person,dog,cat',
    #                     help='segmentation categories (default: person, dog, cat)')

    parser.add_argument('--dataset', type=str, default='pascal_voc', metavar='N',
                        choices=['coco', 'pascal_voc'],
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='/home/chaoyanghe/BruteForce/FedML/data/pascal_voc',
                        help='data directory (default = /home/chaoyanghe/BruteForce/FedML/data/pascal_voc)')
 
    parser.add_argument('--checkname', type=str, default='deeplab-resnet-os8', help='set the checkpoint name')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--client_num_in_total', type=int, default=3, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=3, metavar='NN',
                        help='number of workers')

    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 32)')

    parser.add_argument('--sync_bn', type=bool, default=False,
                        help='whether to use sync bn (default: auto)')

    parser.add_argument('--freeze_bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='adam')

    parser.add_argument('--lr', type=float, default=0.007, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')

    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')

    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')

    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')

    parser.add_argument('--loss_type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')

    parser.add_argument('--epochs', type=int, default=2, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=200,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu_server_num', type=int, default=1,
                        help='gpu_server_num')

    parser.add_argument('--gpu_num_per_server', type=int, default=4,
                        help='gpu_num_per_server')


    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    args = parser.parse_args()

    return args

    ### Args to add ###
    # lr_scheduler
    # outstride
    # freeze_bn
    # sync_bn
    # categories
    # backbone
    # backbone-pretrained


def load_data(process_id, args, dataset_name):
    if dataset_name == "coco":
        data_loader = load_partition_data_coco
    elif dataset_name == "pascal_voc":
        data_loader = load_partition_data_pascal_voc
    train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
    train_data_local_dict, test_data_local_dict, class_num = data_loader(args.dataset, args.data_dir, args.partition_method, args.partition_alpha,
        args.client_num_in_total, args.batch_size)

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict,
    train_data_local_dict, test_data_local_dict, class_num]

    # train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num = data_loader(
    #     process_id, args.dataset, args.data_dir, args.partition_method, args.partition_alpha,
    #     args.client_num_in_total, args.batch_size)
    # dataset = [train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local,
    #            class_num]

    return dataset


def create_model(args, model_name, output_dim, img_size = torch.Size([513, 513])):
    if model_name == "deeplab_transformer":
        model = DeeplabTransformer(backbone=args.backbone,
                                   image_size=img_size,
                                   n_classes=output_dim,
                                   output_stride=args.outstride,
                                   pretrained=args.backbone_pretrained,
                                   freeze_bn=args.freeze_bn,
                                   sync_bn=args.sync_bn)


        logging.info('Args.Backbone: {}'.format(args.backbone_freezed))

        if args.backbone_freezed:
            logging.info('Freezing Backbone')
            for param in model.transformer.parameters():
                param.requires_grad = False
        else:
            logging.info('Finetuning Backbone')

        num_params = count_parameters(model)
        logging.info("Deeplab Transformer Model Size = " + str(num_params))
    else:
        raise ('Not Implemented Error')

    return model


def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine, gpu_server_num):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    if process_ID == 0:
        device = torch.device("cuda:" + str(gpu_server_num) if torch.cuda.is_available() else "cpu")
        return device
    process_gpu_dict = dict()
    for client_index in range(fl_worker_num):
        gpu_index = (client_index % gpu_num_per_machine)
        process_gpu_dict[client_index] = gpu_index + gpu_server_num
    logging.info(process_gpu_dict)
    device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    logging.info(device)
    return device


if __name__ == "__main__":
    now = datetime.datetime.now()
    time_start = now.strftime("%Y-%m-%d %H:%M:%S")
    
    logging.info("Executing Image Segmentation at time: {0}".format(time_start))
    
    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    print(args)
    # customize the process name
    str_process_name = "Deeplab-Resnet-Pascal (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    logging.basicConfig(filename='info.log',
                        level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            # project="federated_nas",
            project="fedml",
            name="FedSeg(d)" + str(args.partition_method) + "r" + str(args.comm_round) + "-e" + str(
                args.epochs) + "-lr" + str(
                args.lr),
            config=args
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # GPU arrangement: Please customize this function according your own topology.
    # The GPU server list is configured at "mpi_host_file".
    # If we have 4 machines and each has two GPUs, and your FL network has 8 workers and a central worker.
    # The 4 machines will be assigned as follows:
    # machine 1: worker0, worker4, worker8;
    # machine 2: worker1, worker5;
    # machine 3: worker2, worker6;
    # machine 4: worker3, worker7;
    # Therefore, we can see that workers are assigned according to the order of machine list.
    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server, args.gpu_server_num)

    # load data
    dataset = load_data(process_id, args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict,
     train_data_local_dict, test_data_local_dict, class_num] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=class_num)

    logging.info("Calling FedML_FedSeg_distributed")

    # start "federated averaging (FedAvg)"
    FedML_FedSeg_distributed(process_id, worker_number, device, comm, model, train_data_num, data_local_num_dict,
                             train_data_local_dict, test_data_local_dict, class_num, args)
