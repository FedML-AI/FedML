import argparse
import logging
import os
import random
import socket
import sys
from sklearn.utils import shuffle

import numpy as np
import psutil
import setproctitle
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.model.finance.vfl_classifier import VFLClassifier
from fedml_api.model.finance.vfl_feature_extractor import VFLFeatureExtractor
from fedml_api.data_preprocessing.lending_club_loan.lending_club_dataset import loan_load_three_party_data
from fedml_api.data_preprocessing.NUS_WIDE.nus_wide_dataset import NUS_WIDE_load_three_party_data
from fedml_api.distributed.classical_vertical_fl.vfl_api import FedML_VFL_distributed
from fedml_api.distributed.fedavg.FedAvgAPI import FedML_init


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument('--dataset', type=str, default='lending_club_loan', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--client_number', type=int, default=2, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--comm_round', type=int, default=100,
                        help='how many round of communications we shoud use')

    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--frequency_of_the_test', type=int, default=30,
                        help='the frequency of the algorithms')

    args = parser.parse_args()
    return args


def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    if process_ID == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    process_gpu_dict = dict()
    for client_index in range(fl_worker_num):
        gpu_index = client_index % gpu_num_per_machine
        process_gpu_dict[client_index] = gpu_index

    logging.info(process_gpu_dict)
    device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    logging.info(device)
    return device


if __name__ == "__main__":
    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    # customize the process name
    str_process_name = "Federated Learning:" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(worker_number)
    random.seed(0)

    # GPU management
    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = init_training_device(process_id, worker_number-1, 4)

    # load data
    print("################################ Prepare Data ############################")
    if args.dataset == "lending_club_loan":
        data_dir = "../../../data/lending_club_loan/"
        train, test = loan_load_three_party_data(data_dir)
    elif args.dataset == "NUS_WIDE":
        data_dir = "../../../data/NUS_WIDE"
        class_lbls = ['person', 'animal']
        train, test = NUS_WIDE_load_three_party_data(data_dir, class_lbls, neg_label=0)
    else:
        data_dir = "../../../data/lending_club_loan/"
        train, test = loan_load_three_party_data(data_dir)

    Xa_train, Xb_train, Xc_train, y_train = train
    Xa_test, Xb_test, Xc_test, y_test = test

    Xa_train, Xb_train, Xc_train, y_train = shuffle(Xa_train, Xb_train, Xc_train, y_train)
    Xa_test, Xb_test, Xc_test, y_test = shuffle(Xa_test, Xb_test, Xc_test, y_test)

    train = [Xa_train, Xb_train, Xc_train, y_train]
    test = [Xa_test, Xb_test, Xc_test, y_test]

    guest_data = [Xa_train, y_train, Xa_test, y_test]
    host_data = None
    if process_id == 1:
        host_data = [Xb_train, Xb_test]
    elif process_id == 2:
        host_data = [Xc_train, Xc_test]

    # create models for each worker
    if process_id == 0:
        guest_feature_extractor = VFLFeatureExtractor(input_dim=Xa_train.shape[1], output_dim=10).to(device)
        guest_classifier = VFLClassifier(guest_feature_extractor.get_output_dim(), 1).to(device)
        guest_model = [guest_feature_extractor, guest_classifier]
        host_model = [None, None]
    elif process_id == 1:
        host_feature_extractor = VFLFeatureExtractor(input_dim=Xb_train.shape[1], output_dim=10).to(device)
        host_classifier = VFLClassifier(host_feature_extractor.get_output_dim(), 1).to(device)
        host_model = [host_feature_extractor, host_classifier]
        guest_model = [None, None]
    elif process_id == 2:
        host_feature_extractor = VFLFeatureExtractor(input_dim=Xc_train.shape[1], output_dim=10).to(device)
        host_classifier = VFLClassifier(host_feature_extractor.get_output_dim(), 1).to(device)
        host_model = [host_feature_extractor, host_classifier]
        guest_model = [None, None]
    else:
        guest_model = [None, None]
        host_model = [None, None]

    FedML_VFL_distributed(process_id, worker_number, comm, args, device, guest_data, guest_model, host_data, host_model)
