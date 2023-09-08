import copy
import logging
import random
import time
import numpy as np
import torch
import wandb

from fedml import mlops

from .utils import transform_list_to_tensor
from ....core.security.fedml_attacker import FedMLAttacker
from ....core.security.fedml_defender import FedMLDefender

class FedAVGAggregator(object):
    """
        Represents a Federated Averaging (FedAVG) aggregator for federated learning.

        Args:
            train_global: The global training dataset.
            test_global: The global testing dataset.
            all_train_data_num: The total number of training data samples.
            train_data_local_dict: A dictionary mapping worker indices to their local training datasets.
            test_data_local_dict: A dictionary mapping worker indices to their local testing datasets.
            train_data_local_num_dict: A dictionary mapping worker indices to the number of local training samples.
            worker_num: The number of worker nodes participating in the federated learning.
            device: The device (e.g., 'cuda' or 'cpu') used for computations.
            args: Additional configuration arguments.
            server_aggregator: The server-side aggregator used for communication with workers.
    """
    def __init__(
        self,
        train_global,
        test_global,
        all_train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        worker_num,
        device,
        args,
        server_aggregator,
    ):
        self.aggregator = server_aggregator

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        """
        Get the global model parameters from the aggregator.

        Returns:
            dict: The global model parameters.
        """
        return self.aggregator.get_model_params()

    def set_global_model_params(self, model_parameters):
        """
        Set the global model parameters in the aggregator.

        Args:
            model_parameters (dict): The global model parameters to set.
        """
        self.aggregator.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        """
        Add the locally trained model result from a worker.

        Args:
            index: The index or ID of the worker.
            model_params (dict): The model parameters trained by the worker.
            sample_num (int): The number of training samples used by the worker.
        """
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        """
        Check if model results have been received from all workers.

        Returns:
            bool: True if results have been received from all workers, False otherwise.
        """
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        """
        Aggregate the model updates from worker nodes using Federated Averaging (FedAVG).

        Returns:
            dict: The averaged model parameters.
        """
        start_time = time.time()
        model_list = []

        for idx in range(self.worker_num):
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            # training_num += self.sample_num_dict[idx]
        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        if FedMLAttacker.get_instance().is_model_attack():
            model_list = FedMLAttacker.get_instance().attack_model(raw_client_grad_list=model_list, extra_auxiliary_info=None)

        if FedMLDefender.get_instance().is_defense_enabled():
            # todo: update extra_auxiliary_info according to defense type
            averaged_params = FedMLDefender.get_instance().defend(
                raw_client_grad_list=model_list,
                base_aggregation_func=self._fedavg_aggregation_,
                extra_auxiliary_info=self.get_global_model_params(),
            )
        else:
            averaged_params = self._fedavg_aggregation_(model_list)

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def _fedavg_aggregation_(self, model_list):
        """
        Perform the FedAVG aggregation on a list of local model updates.

        Args:
            model_list (list): A list of tuples containing local sample numbers and model parameters.

        Returns:
            dict: The aggregated model parameters.
        """
        training_num = 0
        for i in range(0, len(model_list)):
            local_sample_number, local_model_params = model_list[i]
            training_num += local_sample_number
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                if i == 0:
                    averaged_params[k] = (
                        local_model_params[k] * local_sample_number / training_num
                    )
                else:
                    averaged_params[k] += (
                        local_model_params[k] * local_sample_number / training_num
                    )
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        """
        Randomly sample a subset of clients for a federated learning round.

        Args:
            round_idx (int): The index of the current federated learning round.
            client_num_in_total (int): The total number of clients available.
            client_num_per_round (int): The number of clients to sample for the current round.

        Returns:
            list: A list of client indexes selected for the current round.
        """
        if client_num_in_total == client_num_per_round:
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                range(client_num_in_total), num_clients, replace=False
            )
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes


    def _generate_validation_set(self, num_samples=10000):
        """
        Generate a validation set for testing purposes.

        Args:
            num_samples (int): The number of samples to include in the validation set.

        Returns:
            DataLoader: A DataLoader containing the validation set.
        """
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num = len(self.test_global.dataset)
            sample_indices = random.sample(
                range(test_data_num), min(num_samples, test_data_num)
            )
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(
                subset, batch_size=self.args.batch_size
            )
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx):
        """
        Perform testing on the server for all clients in a federated learning round.

        Args:
            round_idx (int): The index of the current federated learning round.
        """
        if self.aggregator.test_all(
            self.train_data_local_dict,
            self.test_data_local_dict,
            self.device,
            self.args,
        ):
            return

        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))
            self.aggregator.test_all(
                self.train_data_local_dict,
                self.test_data_local_dict,
                self.device,
                self.args,
            )

            if round_idx == self.args.comm_round - 1:
                # we allow to return four metrics, such as accuracy, AUC, loss, etc.
                metric_result_in_current_round = self.aggregator.test(self.test_global, self.device, self.args)
            else:
                metric_result_in_current_round = self.aggregator.test(self.val_global, self.device, self.args)
            logging.info("metric_result_in_current_round = {}".format(metric_result_in_current_round))
        else:
            mlops.log({"round_idx": round_idx})
