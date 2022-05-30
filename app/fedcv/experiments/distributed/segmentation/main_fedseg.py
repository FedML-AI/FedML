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

from FedML.fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file
from FedML.fedml_api.distributed.fedseg.FedSegAPI import FedML_init, FedML_FedSeg_distributed
from FedML.fedml_api.distributed.fedseg.utils import count_parameters

#from data_preprocessing.coco.segmentation.data_loader.py import load_partition_data_distributed_coco_segmentation, load_partition_data_coco_segmentation
from data_preprocessing.pascal_voc_augmented.data_loader import load_partition_data_distributed_pascal_voc, \
    load_partition_data_pascal_voc
from data_preprocessing.cityscapes.data_loader import load_partition_data_distributed_cityscapes, \
    load_partition_data_cityscapes
from model.segmentation.deeplabV3_plus import DeepLabV3_plus
from model.segmentation.unet import UNet
from training.segmentation_trainer import SegmentationTrainer


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--process_name', type=str, default='FedSeg-distributed:',
                        help='Machine process names')

    parser.add_argument('--model', type=str, default='deeplabV3_plus', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--backbone', type=str, default='resnet',
                        help='employ with backbone (default: xception)')

    parser.add_argument('--backbone_pretrained', type=str2bool, nargs='?', const=True, default=True,
                        help='pretrained backbone (default: True)')

    parser.add_argument('--backbone_freezed', type=str2bool, nargs='?', const=True, default=False,
                        help='Freeze backbone to extract features only once (default: False)')

    parser.add_argument('--extract_feat', type=str2bool, nargs='?', const=True, default=False,
                        help='Extract Feature Maps of (default: False) NOTE: --backbone_freezed has to be True for this argument to be considered')

    parser.add_argument('--outstride', type=int, default=16,
                        help='network output stride (default: 16)')

    parser.add_argument('--dataset', type=str, default='pascal_voc', metavar='N',
                        choices=['coco', 'pascal_voc', 'cityscapes'],
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='/home/chaoyanghe/BruteForce/FedML/data/pascal_voc',
                        help='data directory (default = /home/chaoyanghe/BruteForce/FedML/data/pascal_voc)')
 
    parser.add_argument('--checkname', type=str, default='deeplab-resnet-finetune-hetero', help='set the checkpoint name')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--client_num_in_total', type=int, default=3, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=3, metavar='NN',
                        help='number of workers')

    parser.add_argument('--save_client_model', type=str2bool, nargs='?', const=True, default=False,
                        help='whether to save locally trained model by clients (default: False')

    parser.add_argument('--save_model', type=str2bool, nargs='?', const=True, default=False,
                        help='whether to save best averaged model (default: False')

    parser.add_argument('--load_model', type=str2bool, nargs='?', const=True, default=False,
                        help='whether to load pre-trained model weights (default: False')

    parser.add_argument('--model_path', type=str, default=None,
                        help='Pre-trained saved model path  NOTE: --load has to be True for this argument to be considered')

    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 32)')

    parser.add_argument('--sync_bn', type=str2bool, nargs='?', const=True, default=False,
                        help='whether to use sync bn (default: False)')

    parser.add_argument('--freeze_bn', type=str2bool, nargs='?', const=True, default=False,
                        help='whether to freeze bn parameters (default: False)')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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

    parser.add_argument('--evaluation_frequency', type=int, default=5,
                        help='Frequency of model evaluation on training dataset (Default: every 5th round)')

    parser.add_argument('--gpu_server_num', type=int, default=1,
                        help='gpu_server_num')
    
    parser.add_argument('--gpu_num_per_server', type=int, default=4,
                        help='gpu_num_per_server')

    parser.add_argument('--gpu_mapping_file', type=str, default="gpu_mapping.yaml",
                        help='the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.')

    parser.add_argument('--gpu_mapping_key', type=str, default="mapping_config1_5",
                        help='the key in gpu utilization file')

    parser.add_argument('--image_size', type=int, default=512,
                        help='Specifies the input size of the model (transformations are applied to scale or crop the image)')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    args = parser.parse_args()

    return args


def load_data(process_id, args, dataset_name):
    data_loader = None
    if dataset_name == "coco":
        pass
       # data_loader = load_partition_data_coco
    elif dataset_name == "pascal_voc":
        data_loader = load_partition_data_pascal_voc
    elif dataset_name == 'cityscapes':
        data_loader = load_partition_data_cityscapes
    train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
    train_data_local_dict, test_data_local_dict, class_num = data_loader(args.dataset, args.data_dir, args.partition_method, args.partition_alpha,
        args.client_num_in_total, args.batch_size, args.image_size)

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict,
    train_data_local_dict, test_data_local_dict, class_num]

    return dataset


def create_model(args, model_name, output_dim, img_size):
    if model_name == "deeplabV3_plus":
        model = DeepLabV3_plus(backbone=args.backbone,
                          image_size=img_size,
                          n_classes=output_dim,
                          output_stride=args.outstride,
                          pretrained=args.backbone_pretrained,
                          freeze_bn=args.freeze_bn,
                          sync_bn=args.sync_bn)

        if args.backbone_freezed:
            logging.info('Freezing Backbone')
            for param in model.feature_extractor.parameters():
                param.requires_grad = False
        else:
            logging.info('Finetuning Backbone')

        num_params = count_parameters(model)
        logging.info("DeepLabV3_plus Model Size : {}".format(num_params))
    

    elif model_name == "unet":
        model = UNet(backbone=args.backbone,
                     output_stride=args.outstride,
                     n_classes=output_dim,
                     pretrained=args.backbone_pretrained,
                     sync_bn=args.sync_bn)

        num_params = count_parameters(model)
        logging.info("Unet Model Size : {}".format(num_params))
        

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
    
    device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    logging.info('GPU process allocation {0}'.format(process_gpu_dict))
    logging.info('GPU device available {0}'.format(device))
    return device


if __name__ == "__main__":

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # customize the log format
    logging.basicConfig(filename='info.log',
                        level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    now = datetime.datetime.now()
    time_start = now.strftime("%Y-%m-%d %H:%M:%S")    
    logging.info("Executing Image Segmentation at time: {0}".format(time_start))

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info('Given arguments {0}'.format(args))

    # customize the process name
    str_process_name = args.process_name + str(process_id)
    setproctitle.setproctitle(str_process_name)

    hostname = socket.gethostname()
    logging.info("Host and process details")
    logging.info("process ID: {0}, host name: {1}, process ID: {2}, process name: {3}, worker number: {4}".format(process_id,hostname,os.getpid(), psutil.Process(os.getpid()), worker_number))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            project = "fedcv-segmentation",
            name = args.process_name + str(args.partition_method) + "r" + str(args.comm_round) + "-e" + str(
                args.epochs) + "-lr" + str(
                args.lr),
            config = args
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
   
    device = mapping_processes_to_gpu_device_from_yaml_file(process_id, worker_number, args.gpu_mapping_file, args.gpu_mapping_key)
    #device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server, args.gpu_server_num)

    # load data
    dataset = load_data(process_id, args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict,
     train_data_local_dict, test_data_local_dict, class_num] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=class_num, img_size=torch.Size([args.image_size, args.image_size]))

    if args.load_model:
        
        try:
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['state_dict'])
        except:
            raise("Failed to load pre-trained model")
        
    # define my own trainer
    model_trainer = SegmentationTrainer(model, args)

    logging.info("Calling FedML_FedSeg_distributed")

    FedML_FedSeg_distributed(process_id, worker_number, device, comm, model, train_data_num, data_local_num_dict,
                             train_data_local_dict, test_data_local_dict, args, model_trainer)
