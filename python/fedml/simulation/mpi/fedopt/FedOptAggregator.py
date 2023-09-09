import copy
import logging
import random
import time

import numpy as np
import torch
import wandb

from .optrepo import OptRepo
from .utils import transform_list_to_tensor


class FedOptAggregator(object):
    """Aggregator for Federated Optimization.

    This class manages the aggregation of model updates from client devices in a federated optimization setting.

    Args:
        train_global: The global training dataset.
        test_global: The global testing dataset.
        all_train_data_num: The total number of training samples across all clients.
        train_data_local_dict: A dictionary mapping client indices to their local training datasets.
        test_data_local_dict: A dictionary mapping client indices to their local testing datasets.
        train_data_local_num_dict: A dictionary mapping client indices to the number of samples in their local training datasets.
        worker_num: The number of worker (client) devices.
        device: The device (CPU or GPU) to use for model aggregation.
        args: An argparse.Namespace object containing various configuration options.
        server_aggregator: An optional ServerAggregator object used for model aggregation.

    Attributes:
        aggregator: The server aggregator for model aggregation.
        args: An argparse.Namespace object containing various configuration options.
        train_global: The global training dataset.
        test_global: The global testing dataset.
        val_global: A subset of the testing dataset used for validation.
        all_train_data_num: The total number of training samples across all clients.
        train_data_local_dict: A dictionary mapping client indices to their local training datasets.
        test_data_local_dict: A dictionary mapping client indices to their local testing datasets.
        train_data_local_num_dict: A dictionary mapping client indices to the number of samples in their local training datasets.
        worker_num: The number of worker (client) devices.
        device: The device (CPU or GPU) to use for model aggregation.
        model_dict: A dictionary mapping client indices to their local model updates.
        sample_num_dict: A dictionary mapping client indices to the number of samples used for their local updates.
        flag_client_model_uploaded_dict: A dictionary tracking whether each client has uploaded its local model update.
        opt: The server optimizer used for model aggregation.
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
        """Initialize the FedOptAggregator.

        Args:
            train_global: Global training data.
            test_global: Global test data.
            all_train_data_num: Total number of training data samples.
            train_data_local_dict: Dictionary of local training data.
            test_data_local_dict: Dictionary of local test data.
            train_data_local_num_dict: Dictionary of the number of local training data samples.
            worker_num: Number of worker clients.
            device: The device (e.g., CPU or GPU) for training.
            args: A configuration object containing aggregator parameters.
            server_aggregator: The server aggregator object.
        """
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
        self.opt = self._instantiate_opt()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def _instantiate_opt(self):
        """Instantiate the optimizer.

        Returns:
            torch.optim.Optimizer: The instantiated optimizer.
        """
        return OptRepo.name2cls(self.args.server_optimizer)(
            filter(lambda p: p.requires_grad, self.get_model_params()),
            lr=self.args.server_lr,
            momentum=self.args.server_momentum,
        )

    def get_model_params(self):
        """Get model parameters.

        Returns:
            generator: Generator of model parameters.
        """
        return self.aggregator.model.parameters()

    def get_global_model_params(self):
        """Get global model parameters.

        Returns:
            OrderedDict: Global model parameters.
        """
        return self.aggregator.get_model_params()

    def set_global_model_params(self, model_parameters):
        """Set global model parameters.

        Args:
            model_parameters: New global model parameters.
        """
        self.aggregator.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        """Add locally trained model results.

        Args:
            index: Index of the client.
            model_params: Model parameters.
            sample_num: Number of training samples.
        """
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        """Check if all clients have uploaded their models.

        Returns:
            bool: True if all clients have uploaded their models, False otherwise.
        """
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        """Aggregate locally trained models.

        Returns:
            OrderedDict: Aggregated global model parameters.
        """
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))


        (num0, averaged_params) = model_list[0]

        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # Server optimizer
        self.opt.zero_grad()
        opt_state = self.opt.state_dict()

        self.set_model_global_grads(averaged_params)
        self.opt = self._instantiate_opt()

        self.opt.load_state_dict(opt_state)
        self.opt.step()

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return self.get_global_model_params()

    def set_model_global_grads(self, new_state):
        """Set global model gradients.

        Args:
            new_state: New global model parameters.
        """
        new_model = copy.deepcopy(self.aggregator.model)
        new_model.load_state_dict(new_state)
        with torch.no_grad():
            for parameter, new_parameter in zip(self.aggregator.model.parameters(), new_model.parameters()):
                parameter.grad = parameter.data - new_parameter.data

        model_state_dict = self.aggregator.model.state_dict()
        new_model_state_dict = new_model.state_dict()
        for k in dict(self.aggregator.model.named_parameters()).keys():
            new_model_state_dict[k] = model_state_dict[k]
            
        self.set_global_model_params(new_model_state_dict)

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        """Sample clients for communication.

        Args:
            round_idx: The current communication round.
            client_num_in_total: Total number of clients.
            client_num_per_round: Number of clients to sample per round.

        Returns:
            list: List of sampled client indexes.
        """
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        """Generate a validation dataset.

        Args:
            num_samples: Number of samples in the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx):
        """Test on the server for all clients.

        Args:
            round_idx: The current communication round.
        """
        if self.aggregator.test_all(
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
            logging.info(
                "################test_on_server_for_all_clients : {}".format(round_idx)
            )
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []
            for client_idx in range(self.args.client_num_in_total):
                # Train data
                metrics = self.aggregator.test(
                    self.train_data_local_dict[client_idx], self.device, self.args
                )
                train_tot_correct, train_num_sample, train_loss = (
                    metrics["test_correct"],
                    metrics["test_total"],
                    metrics["test_loss"],
                )
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))

            # Test on training dataset
            train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            train_loss = sum(train_losses) / sum(train_num_samples)
            if self.args.enable_wandb:
                wandb.log({"Train/Acc": train_acc, "round": round_idx})
                wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {"training_acc": train_acc, "training_loss": train_loss}
            logging.info(stats)

            # Test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                metrics = self.aggregator.test(self.test_global, self.device, self.args)
            else:
                metrics = self.aggregator.test(self.val_global, self.device, self.args)

            test_tot_correct, test_num_sample, test_loss = (
                metrics["test_correct"],
                metrics["test_total"],
                metrics["test_loss"],
            )
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # Test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            logging.info(stats)
