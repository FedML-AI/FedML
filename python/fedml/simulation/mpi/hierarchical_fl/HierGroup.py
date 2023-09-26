import logging

from .HierClient import HFLClient
from ...sp.fedavg.fedavg_api import FedAvgAPI


class HierGroup(FedAvgAPI):
    def __init__(
        self,
        idx,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        args,
        device,
        model,
        model_trainer,
    ):
        self.idx = idx
        self.args = args
        self.device = device
        self.client_dict = {}
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model = model
        self.model_trainer = model_trainer
        self.args = args

    def setup_clients(self, total_client_indexes):
        self.client_dict = {}
        for client_idx in total_client_indexes:
            self.client_dict[client_idx] = HFLClient(
                client_idx,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                self.model,
                self.model_trainer,
            )

    def get_sample_number(self, sampled_client_indexes):
        self.group_sample_number = 0
        for client_idx in sampled_client_indexes:
            self.group_sample_number += self.train_data_local_num_dict[client_idx]
        return self.group_sample_number

    def train(self, global_round_idx, w, sampled_client_indexes):
        sampled_client_list = [self.client_dict[client_idx] for client_idx in sampled_client_indexes]
        w_group = w
        w_group_list = []
        for group_round_idx in range(self.args.group_comm_round):
            logging.info("Group ID : {} / Group Communication Round : {}".format(self.idx, group_round_idx))
            w_locals_dict = {}

            # train each client
            for client in sampled_client_list:
                w_local_list = client.train(global_round_idx, group_round_idx, w_group)
                for client_round, w in w_local_list:
                    if not client_round in w_locals_dict:
                        w_locals_dict[client_round] = []
                    w_locals_dict[client_round].append((client.get_sample_number(), w))

            # aggregate local weights
            for client_round in sorted(w_locals_dict.keys()):
                w_locals = w_locals_dict[client_round]
                w_group_list.append((client_round, self._aggregate(w_locals)))

            # update the group weight
            w_group = w_group_list[-1][1]
        return w_group_list
