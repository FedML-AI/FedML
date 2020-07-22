import argparse
import logging
import random

import numpy as np
import wandb

from fedavg_single_process_pytorch.data_loader import DataLoader
from topology_manager import TopologyManager

from client_dsgd import ClientDSGD
from client_pushsum import ClientPushsum

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


switch_wandb = True


def cal_regret(client_list, client_number, t):
    regret = 0
    for client in client_list:
        regret += np.sum(client.get_regret())

    regret = regret / (client_number * (t + 1))
    return regret


def main():
    ROOT_PATH = args.root_path
    RUN_NAME=str(args.mode) + \
             "-id" + str(args.run_id) + \
             "-group_id" + str(args.group_id) + \
             "-n" + str(args.client_number) + \
             "-symm" + str(args.b_symmetric) + \
             "-tu" + str(args.topology_neighbors_num_undirected) + \
             "-td" + str(args.topology_neighbors_num_directed) + \
             "-lr" + str(args.learning_rate),
    if switch_wandb:
        wandb.init(project="decentralized-online-learning",
                   name=str(RUN_NAME),
                   config=args)
    logging.basicConfig(filename="DOL.log",
                        level=logging.INFO,
                        format=' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    logging.info('Decentralized Online Learning.')

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)

    # parameters
    client_number = args.client_number
    iteration_number_T = args.iteration_number
    sample_num_in_total = client_number * iteration_number_T
    beta = args.beta
    input_dim = args.input_dim
    output_dim = args.output_dim
    lr_rate = args.learning_rate
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    topology_neighbors_num_undirected = args.topology_neighbors_num_undirected
    topology_neighbors_num_directed = args.topology_neighbors_num_directed
    latency = args.latency
    b_symmetric = args.b_symmetric
    data_name = args.data_name
    epoch = args.epoch
    time_varying = args.time_varying

    client_id_list = [i for i in range(client_number)]
    print(client_id_list)

    # load data
    data_path = ""
    if data_name == "SUSY":
        data_path = ROOT_PATH + "data/SUSY.csv"
        input_dim = 18
    elif data_name == "RO":
        data_path = ROOT_PATH + "data/datatraining.txt"
        input_dim = 5
    data_loader = DataLoader(data_name, data_path, client_id_list, sample_num_in_total, beta)
    streaming_data = data_loader.load_datastream()

    # create the network topology topology
    if b_symmetric:
        topology_manager = TopologyManager(client_number, True,
                                           undirected_neighbor_num=topology_neighbors_num_undirected)
    else:
        topology_manager = TopologyManager(client_number, False,
                                           undirected_neighbor_num=topology_neighbors_num_undirected,
                                           out_directed_neighbor=topology_neighbors_num_directed)
    topology_manager.generate_topology()

    # create all client instances (each client will create an independent model instance)
    client_list = []
    model_list = []
    for client_id in client_id_list:
        client_data = streaming_data[client_id]
        print("len = " + str(len(client_data)))

        if args.mode == 'PUSHSUM':

            client = ClientPushsum(client_id, client_data, topology_manager, input_dim, output_dim,
                                   iteration_number_T, learning_rate=lr_rate, batch_size=batch_size,
                                   weight_decay=weight_decay, latency=latency, b_symmetric=b_symmetric,
                                   time_varying=time_varying)

        elif args.mode == 'DOL':

            client = ClientDSGD(client_id, client_data, topology_manager, input_dim, output_dim,
                        iteration_number_T, learning_rate=lr_rate, batch_size=batch_size,
                                weight_decay=weight_decay, latency=latency, b_symmetric=b_symmetric)

        else:
            client = ClientDSGD(client_id, client_data, topology_manager, input_dim, output_dim,
                        iteration_number_T, learning_rate=lr_rate, batch_size=batch_size,
                                weight_decay=weight_decay, latency=latency, b_symmetric=b_symmetric)
            print("do nothing...")

        client_list.append(client)
        # model_list.append(client.model_z)

    if switch_wandb:
        wandb.watch(model_list, log="all")

    LOG_FILE_NAME=RUN_NAME
    f_log = open(ROOT_PATH + "log/%s.txt" % LOG_FILE_NAME, mode ='w+', encoding='utf-8')

    for t in range(iteration_number_T * epoch):
        logging.info('--- Iteration %d ---' % t)

        if args.mode == 'DOL' or args.mode == 'PUSHSUM':
            for client in client_list:
                # line 4: Locally computes the intermedia variable
                client.train(t)

                # line 5: send to neighbors
                client.send_local_gradient_to_neighbor(client_list)

            # line 6: update
            for client in client_list:
                client.update_local_parameters()
        else:
            for client in client_list:
                client.train_local(t)

        regret = cal_regret(client_list, client_number, t)
        print("regret = %s" % regret)

        if switch_wandb:
            wandb.log({"Average Loss": regret, "iteration": t})

        f_log.write("%f,%f\n" % (t, regret))

    f_log.close()
    if switch_wandb:
        wandb.save(str(LOG_FILE_NAME))


if __name__ == '__main__':
    main()
