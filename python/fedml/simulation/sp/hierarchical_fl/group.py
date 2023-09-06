import logging

from .client import HFLClient
from ..fedavg.fedavg_api import FedAvgAPI


class Group(FedAvgAPI):
    """
    Represents a group of clients in a federated learning setting.

    Args:
        idx (int): Index of the group.
        total_client_indexes (list): List of client indexes in the group.
        train_data_local_dict: Dictionary containing local training data for each client.
        test_data_local_dict: Dictionary containing local test data for each client.
        train_data_local_num_dict: Dictionary containing the number of local training samples for each client.
        args: Arguments for group configuration.
        device: Device (e.g., 'cuda' or 'cpu') to perform computations.
        model: The shared model used by clients in the group.
        model_trainer: Trainer for the shared model.
    """
    def __init__(
        self,
        idx,
        total_client_indexes,
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
        for client_idx in total_client_indexes:
            self.client_dict[client_idx] = HFLClient(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                args,
                device,
                model,
                model_trainer,
            )

    def get_sample_number(self, sampled_client_indexes):
        """
        Calculate the total number of training samples in the group.

        Args:
            sampled_client_indexes (list): List of sampled client indexes.

        Returns:
            int: Total number of training samples in the group.
        """
        self.group_sample_number = 0
        for client_idx in sampled_client_indexes:
            self.group_sample_number += self.train_data_local_num_dict[client_idx]
        return self.group_sample_number

    def train(self, global_round_idx, w, sampled_client_indexes):
        """
        Train the group of clients using federated learning.

        Args:
            global_round_idx (int): Global round index.
            w: Model weights to initialize training.
            sampled_client_indexes (list): List of sampled client indexes.

        Returns:
            list: A list of tuples containing global epoch and aggregated model weights.
        """
        sampled_client_list = [self.client_dict[client_idx] for client_idx in sampled_client_indexes]
        w_group = w
        w_group_list = []
        for group_round_idx in range(self.args.group_comm_round):
            logging.info("Group ID : {} / Group Communication Round : {}".format(self.idx, group_round_idx))
            w_locals_dict = {}

            # train each client
            for client in sampled_client_list:
                w_local_list = client.train(global_round_idx, group_round_idx, w_group)
                for global_epoch, w in w_local_list:
                    if not global_epoch in w_locals_dict:
                        w_locals_dict[global_epoch] = []
                    w_locals_dict[global_epoch].append((client.get_sample_number(), w))

            # aggregate local weights
            for global_epoch in sorted(w_locals_dict.keys()):
                w_locals = w_locals_dict[global_epoch]
                w_group_list.append((global_epoch, self._aggregate(w_locals)))

            # update the group weight
            w_group = w_group_list[-1][1]
        return w_group_list
