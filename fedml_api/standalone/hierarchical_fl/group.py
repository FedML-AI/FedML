import logging

from fedml_api.standalone.hierarchical_fl.client import Client
from fedml_api.standalone.fedavg.fedavg_trainer import FedAvgTrainer

class Group(FedAvgTrainer):

    def __init__(self, idx, total_client_indexes, train_data_local_dict, test_data_local_dict, train_data_local_num_dict, args, device, model):
        self.idx = idx
        self.args = args
        self.device = device
        self.client_dict = {}
        self.group_sample_number = 0
        for client_idx in total_client_indexes:
            self.client_dict[client_idx] = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], args, device, model)
            self.group_sample_number += train_data_local_num_dict[client_idx]

    def get_sample_number(self):
        return self.group_sample_number

    def train(self, w, sampled_client_indexes):
        sampled_client_list = [self.client_dict[client_idx] for client_idx in sampled_client_indexes]
        w_group = w
        w_group_list = []
        for group_round_idx in range(self.args.group_comm_round):
            logging.info("Group ID : {} / Group Communication Round : {}".format(self.idx, group_round_idx))
            w_locals_list = [[] for _ in range(self.args.epochs)]
            for client in sampled_client_list:
                # train on new dataset
                w_local_list = client.train(w_group)
                for epoch, w in enumerate(w_local_list):
                    w_locals_list[epoch].append((client.get_sample_number(), w))

            # aggregate local weights
            for w_locals in w_locals_list:
                w_group_list.append(self.aggregate(w_locals))

            # update last weight
            w_group = w_group_list[-1]
        return w_group_list
