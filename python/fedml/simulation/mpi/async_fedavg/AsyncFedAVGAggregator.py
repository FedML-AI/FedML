import copy
import logging
import random
import numpy as np
import torch
import wandb
import collections

from .utils import transform_list_to_tensor
from ....core.security.fedml_defender import FedMLDefender

from ....core.schedule.runtime_estimate import t_sample_fit

class AsyncFedAVGAggregator(object):
    """
    Aggregator for the asynchronous Federated Averaging server in a federated learning system.

    Args:
        train_global: Global training data.
        test_global: Global testing data.
        all_train_data_num: Total number of training data samples.
        train_data_local_dict: Dictionary containing local training data for each client.
        test_data_local_dict: Dictionary containing local testing data for each client.
        train_data_local_num_dict: Dictionary containing the number of local training data samples for each client.
        worker_num: Number of worker processes.
        device: The computing device (e.g., CPU or GPU).
        args: Command-line arguments and configurations.
        model_trainer: Trainer for the federated learning model.

    Attributes:
        trainer: Trainer for the federated learning model.
        args: Command-line arguments and configurations.
        train_global: Global training data.
        test_global: Global testing data.
        val_global: Global validation data generated from the global training data.
        all_train_data_num: Total number of training data samples.
        train_data_local_dict: Dictionary containing local training data for each client.
        test_data_local_dict: Dictionary containing local testing data for each client.
        train_data_local_num_dict: Dictionary containing the number of local training data samples for each client.
        worker_num: Number of worker processes.
        device: The computing device (e.g., CPU or GPU).
        model_dict: Dictionary containing client models indexed by client ID.
        sample_num_dict: Dictionary containing the number of samples trained by each client.
        flag_client_model_uploaded_dict: Dictionary tracking whether client models have been uploaded.
        runtime_history: Dictionary containing runtime information for clients.
        model_weights: Global model weights updated during aggregation.
        client_running_status: Array tracking the status of running clients.

    Methods:
        get_global_model_params():
            Get the global model parameters.

        set_global_model_params(model_parameters):
            Set the global model parameters.

        add_local_trained_result(index, model_params, local_sample_number,
            current_round, client_round):
            Add the locally trained model results to the aggregator and update the global model.

        client_schedule(round_idx, client_indexes, mode="simulate"):
            Generate a schedule for clients based on runtime information.

        get_average_weight(client_indexes):
            Calculate the average weight assigned to each client based on the number of training samples.

        client_sampling(round_idx, client_num_in_total, client_num_per_round):
            Sample clients for communication in a round.

        _generate_validation_set(num_samples=10000):
            Generate a validation set from the global testing data.

        test_on_server_for_all_clients(round_idx):
            Perform testing on the server for all clients and log the results.
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
        model_trainer,
    ):
        self.trainer = model_trainer

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
        self.runtime_history = {}
        for i in range(self.worker_num):
            self.runtime_history[i] = {}
            for j in range(self.args.client_num_in_total):
                self.runtime_history[i][j] = []
        self.model_weights = self.trainer.get_model_params()
        self.client_running_status = np.array([])


    def get_global_model_params(self):
        """
        Get the global model parameters.

        Returns:
            dict: Global model parameters.
        """
        # return self.trainer.get_model_params()
        return self.model_weights

    def set_global_model_params(self, model_parameters):
        """
        Set the global model parameters.

        Args:
            model_parameters (dict): Global model parameters to be set.

        Returns:
            None
        """
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, local_sample_number,
            current_round, client_round):
        """
        Add the locally trained model results to the aggregator and update the global model.

        Args:
            index (int): Index of the client.
            model_params (dict): Model parameters trained by the client.
            local_sample_number (int): Number of local training data samples used by the client.
            current_round (int): Current communication round.
            client_round (int): Round index for the client.

        Returns:
            None
        """
        logging.info("add_model. index = %d" % index)

        self.client_running_status = np.setdiff1d(self.client_running_status,
                    np.array([index]), assume_unique=False) 
        client_staleness = current_round - client_round
        weight = 1. / (1 + client_staleness)

        self.model_dict[index] = model_params
        # self.sample_num_dict[index] = sample_num
        for name, param in self.model_weights.items():
            self.model_weights[name] += model_params[name] * weight


    def client_schedule(self, round_idx, client_indexes, mode="simulate"):
        """
        Generate a schedule for clients based on runtime information.

        Args:
            round_idx (int): Current communication round.
            client_indexes (list): List of client indexes.
            mode (str): The scheduling mode ("simulate" or "release").

        Returns:
            list: List of client schedules.
        """
        self.runtime_history = {}
        for i in range(self.worker_num):
            self.runtime_history[i] = {}
            for j in range(self.args.client_num_in_total):
                self.runtime_history[i][j] = []

        fit_params, fit_funcs, fit_errors = t_sample_fit(
            self.worker_num, self.args.client_num_in_total, self.runtime_history, 
            self.train_data_local_num_dict, uniform_client=True, uniform_gpu=False)

        client_schedule = np.array_split(client_indexes, self.worker_num)
        return client_schedule


    def get_average_weight(self, client_indexes):
        """
        Calculate the average weight assigned to each client based on the number of training samples.

        Args:
            client_indexes (list): List of client indexes.

        Returns:
            dict: A dictionary mapping client indexes to their respective average weights.
        """
        average_weight_dict = {}
        training_num = 0
        for client_index in client_indexes:
            training_num += self.train_data_local_num_dict[client_index]

        for client_index in client_indexes:
            average_weight_dict[client_index] = self.train_data_local_num_dict[client_index] / training_num
        return average_weight_dict


    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        """
        Sample clients for communication in a round.

        Args:
            round_idx (int): Current communication round.
            client_num_in_total (int): Total number of clients.
            client_num_per_round (int): Number of clients to sample per round.

        Returns:
            list: List of client indexes selected for communication in the current round.
        """
        num_clients = min(client_num_per_round, client_num_in_total)
        np.random.seed(
            round_idx
        )  # make sure for each comparison, we are selecting the same clients each round
        sleep_clients = np.setdiff1d(np.array(range(client_num_in_total)),
                    self.client_running_status, assume_unique=False) 
        client_indexes = np.random.choice(
            sleep_clients, 1, replace=False
        )
        logging.info("client_indexes = %s" % str(client_indexes))
        self.client_running_status = np.concatenate((self.client_running_status, client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        """
        Generate a validation set from the global testing data.

        Args:
            num_samples (int): Number of samples to include in the validation set.

        Returns:
            torch.utils.data.DataLoader: DataLoader containing the validation set.
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
        Perform testing on the server for all clients and log the results.

        Args:
            round_idx (int): Current communication round.

        Returns:
            None
        """
        if self.trainer.test_on_the_server(
            self.train_data_local_dict,
            self.test_data_local_dict,
            self.device,
            self.args,
        ):
            return

        if (
            round_idx % self.args.frequency_of_the_test == 0
            or round_idx == self.args.comm_round - 1
        ):
            self.set_global_model_params(self.model_weights)
            logging.info(
                "################test_on_server_for_all_clients : {}".format(round_idx)
            )
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []
            # for client_idx in range(self.args.client_num_in_total):
            #     # train data
            #     metrics = self.trainer.test(
            #         self.train_data_local_dict[client_idx], self.device, self.args
            #     )
            #     train_tot_correct, train_num_sample, train_loss = (
            #         metrics["test_correct"],
            #         metrics["test_total"],
            #         metrics["test_loss"],
            #     )
            #     train_tot_corrects.append(copy.deepcopy(train_tot_correct))
            #     train_num_samples.append(copy.deepcopy(train_num_sample))
            #     train_losses.append(copy.deepcopy(train_loss))

            # test on training dataset
            # train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            # train_loss = sum(train_losses) / sum(train_num_samples)
            # if self.args.enable_wandb:
            #     wandb.log({"Train/Acc": train_acc, "round": round_idx})
            #     wandb.log({"Train/Loss": train_loss, "round": round_idx})
            # stats = {"training_acc": train_acc, "training_loss": train_loss}
            # logging.info(stats)

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)

            test_tot_correct, test_num_sample, test_loss = (
                metrics["test_correct"],
                metrics["test_total"],
                metrics["test_loss"],
            )
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            logging.info(stats)
