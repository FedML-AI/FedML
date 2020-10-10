import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.standalone.decentralized.decentralized_fl_api import FedML_decentralized_fl
from fedml_api.data_preprocessing.UCI.data_loader_for_susy_and_ro import DataLoader

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--root_path', type=str, default="./", help='set the root path')
parser.add_argument('--mode', type=str, default="DOL", help='LOCAL; DOL; COL')
parser.add_argument('--iteration_number', type=int, default=2000, help='iteration_number')
parser.add_argument('--beta', type=float, default=0, help='beta: 0; 0.2; 0.5')
parser.add_argument('--input_dim', type=float, default=18)
parser.add_argument('--output_dim', type=float, default=1)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=1)

parser.add_argument('--data_name', type=str, default="SUSY", help='SUSY; RO')
parser.add_argument('--epoch', type=int, default=1, help='1,2,3,4,5')

parser.add_argument('--run_id', type=int, default=0)
parser.add_argument('--group_id', type=int, default=0)
parser.add_argument('--client_number', type=int, default=15, help='client number')  # network size
parser.add_argument('--b_symmetric', type=int, default=0)
parser.add_argument('--topology_neighbors_num_undirected', type=int, default=4)
parser.add_argument('--topology_neighbors_num_directed', type=int, default=4)
parser.add_argument('--latency', type=float, default=0)
parser.add_argument('--time_varying', type=int, default=0)
args = parser.parse_args()


def main():
    ROOT_PATH = args.root_path
    RUN_NAME = str(args.mode) + \
               "-id" + str(args.run_id) + \
               "-group_id" + str(args.group_id) + \
               "-n" + str(args.client_number) + \
               "-symm" + str(args.b_symmetric) + \
               "-tu" + str(args.topology_neighbors_num_undirected) + \
               "-td" + str(args.topology_neighbors_num_directed) + \
               "-lr" + str(args.learning_rate),

    wandb.init(project="fedml",
               name=str(RUN_NAME),
               config=args)

    logging.basicConfig(level=logging.INFO,
                        format=' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    logging.info('Decentralized Online Learning.')

    # fix random seeds
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # create id list
    client_number = args.client_number
    client_id_list = [i for i in range(client_number)]
    print(client_id_list)

    # load data
    iteration_number_T = args.iteration_number
    sample_num_in_total = client_number * iteration_number_T
    beta = args.beta

    data_name = args.data_name
    data_path = ""
    if data_name == "SUSY":
        data_path = ROOT_PATH + "SUSY/SUSY.csv"
        input_dim = 18
    elif data_name == "RO":
        data_path = ROOT_PATH + "room_occupancy/datatraining.txt"
        input_dim = 5
    else:
        input_dim = 5
    data_loader = DataLoader(data_name, data_path, client_id_list, sample_num_in_total, beta)
    streaming_data = data_loader.load_datastream()

    # create model
    model = LogisticRegression(input_dim, args.output_dim)
    model_cache = LogisticRegression(input_dim, args.output_dim)

    # start training
    FedML_decentralized_fl(client_number, client_id_list, streaming_data, model, model_cache, args)


if __name__ == '__main__':
    main()
