import copy
import logging
import random
import time

import numpy as np
import torch
import wandb

from .optrepo import OptRepo
from .utils import transform_list_to_tensor

from ....core.schedule.seq_train_scheduler import SeqTrainScheduler
from ....core.schedule.runtime_estimate import t_sample_fit


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
        runtime_history: A dictionary to track the runtime history of clients.
        runtime_avg: A dictionary to track the average runtime of clients.
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
        """Instantiate the server optimizer based on configuration options.

        Returns:
            torch.optim.Optimizer: The server optimizer.
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
        self.runtime_history = {}
        self.runtime_avg = {}
        for i in range(self.worker_num):
            self.runtime_history[i] = {}
            self.runtime_avg[i] = {}
            for j in range(self.args.client_num_in_total):
                self.runtime_history[i][j] = []
                self.runtime_avg[i][j] = None


    def _instantiate_opt(self):
        """
        Instantiate the server optimizer based on configuration options.

        Returns:
            torch.optim.Optimizer: The server optimizer.
        """
        return OptRepo.name2cls(self.args.server_optimizer)(
            filter(lambda p: p.requires_grad, self.get_model_params()),
            lr=self.args.server_lr,
            momentum=self.args.server_momentum,
        )

    def get_model_params(self):
        """
        Get the model parameters in the form of a generator.

        Returns:
            generator: A generator of model parameters.
        """
        # return model parameters in type of generator
        return self.aggregator.model.parameters()

    def get_global_model_params(self):
        """
        Get the global model parameters as an ordered dictionary.

        Returns:
            collections.OrderedDict: The global model parameters.
        """

        # return model parameters in type of ordered_dict
        return self.aggregator.get_model_params()

    def set_global_model_params(self, model_parameters):
        """
        Set the global model parameters based on a provided dictionary.

        Args:
            model_parameters (dict): A dictionary containing global model parameters.
        """
        self.aggregator.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params):
        """
        Add the local trained model update for a client.

        Args:
            index (int): The index of the client.
            model_params (dict): The model parameters of the local trained model.
        """
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        # self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        """
        Check whether all clients have uploaded their local model updates.

        Returns:
            bool: True if all clients have uploaded their updates, False otherwise.
        """
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True



    def workload_estimate(self, client_indexes, mode="simulate"):
        """
        Estimate the workload for selected clients.

        Args:
            client_indexes (list): The indices of selected clients.
            mode (str): The mode for workload estimation ("simulate" or "real").

        Returns:
            list: Workload estimates for the selected clients.
        """
        if mode == "simulate":
            client_samples = [
                self.train_data_local_num_dict[client_index]
                for client_index in client_indexes
            ]
            workload = client_samples
        elif mode == "real":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return workload

    def memory_estimate(self, client_indexes, mode="simulate"):
        """
        Estimate the memory requirements for selected clients.

        Args:
            client_indexes (list): The indices of selected clients.
            mode (str): The mode for memory estimation ("simulate" or "real").

        Returns:
            numpy.ndarray: Memory estimates for the selected clients.
        """
        if mode == "simulate":
            memory = np.ones(self.worker_num)
        elif mode == "real":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return memory

    def resource_estimate(self, mode="simulate"):
        """
        Estimate the resource requirements for clients.

        Args:
            mode (str): The mode for resource estimation ("simulate" or "real").

        Returns:
            numpy.ndarray: Resource estimates for clients.
        """
        if mode == "simulate":
            resource = np.ones(self.worker_num)
        elif mode == "real":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return resource

    def record_client_runtime(self, worker_id, client_runtimes):
        """
        Record the runtime of clients.

        Args:
            worker_id (int): The ID of the worker (client).
            client_runtimes (dict): A dictionary mapping client IDs to their runtimes.
        """
        for client_id, runtime in client_runtimes.items():
            self.runtime_history[worker_id][client_id].append(runtime)
        if hasattr(self.args, "runtime_est_mode"):
            if self.args.runtime_est_mode == 'EMA':
                for client_id, runtime in client_runtimes.items():
                    if self.runtime_avg[worker_id][client_id] is None:
                        self.runtime_avg[worker_id][client_id] = runtime
                    else:
                        self.runtime_avg[worker_id][client_id] += self.runtime_avg[worker_id][client_id]/2 + runtime/2
            elif self.args.runtime_est_mode == 'time_window':
                for client_id, runtime in client_runtimes.items():
                    self.runtime_history[worker_id][client_id] = self.runtime_history[worker_id][client_id][-3:]



    def generate_client_schedule(self, round_idx, client_indexes):
        """Generate a schedule of clients for training.

        Args:
            round_idx (int): The current communication round index.
            client_indexes (list): The indices of selected clients.

        Returns:
            list: A schedule of clients for training.
        """
        # self.runtime_history = {}
        # for i in range(self.worker_num):
        #     self.runtime_history[i] = {}
        #     for j in range(self.args.client_num_in_total):
        #         self.runtime_history[i][j] = []
        previous_time = time.time()
        if hasattr(self.args, "simulation_schedule") and round_idx > 5:
            # Need some rounds to record some information. 
            simulation_schedule = self.args.simulation_schedule
            if hasattr(self.args, "runtime_est_mode"):
                if self.args.runtime_est_mode == 'EMA':
                    runtime_to_fit = self.runtime_avg
                elif self.args.runtime_est_mode == 'time_window':
                    runtime_to_fit = self.runtime_history
                else:
                    raise NotImplementedError
            else:
                runtime_to_fit = self.runtime_history

            fit_params, fit_funcs, fit_errors = t_sample_fit(
                self.worker_num, self.args.client_num_in_total, runtime_to_fit, 
                self.train_data_local_num_dict, uniform_client=True, uniform_gpu=False)
            logging.info(f"fit_params: {fit_params}")
            logging.info(f"fit_errors: {fit_errors}")
            avg_fit_error = 0.0
            sum_times = 0
            for gpu, gpu_erros in fit_errors.items():
                for client, client_error in gpu_erros.items():
                    avg_fit_error += client_error
                    sum_times += 1
            avg_fit_error /= sum_times
            if self.args.enable_wandb:
                wandb.log({"RunTimeEstimateError": avg_fit_error, "round": round_idx})

            mode = 0
            workloads = np.array([ self.train_data_local_num_dict[client_id] for client_id in client_indexes])
            constraints = np.array([1]*self.worker_num)
            memory = np.array([100])
            my_scheduler = SeqTrainScheduler(workloads, constraints, memory,
                fit_funcs, uniform_client=True, uniform_gpu=False)
            # my_scheduler = SeqTrainScheduler(workloads, constraints, memory, self.train_data_local_num_dict,
            #     fit_funcs, uniform_client=True, uniform_gpu=False)
            y_schedule, output_schedules = my_scheduler.DP_schedule(mode)
            client_schedule = []
            for indexes in y_schedule:
                client_schedule.append(client_indexes[indexes])
        else:
            client_schedule = np.array_split(client_indexes, self.worker_num)
        if self.args.enable_wandb:
            wandb.log({"RunTimeSchedule": time.time() - previous_time, "round": round_idx})
        logging.info(f"Schedules: {client_schedule}")
        return client_schedule


    def get_average_weight(self, client_indexes):
        """Calculate the average weight for selected clients.

        Args:
            client_indexes (list): The indices of selected clients.

        Returns:
            dict: A dictionary mapping client indices to their average weights.
        """
        average_weight_dict = {}
        training_num = 0
        for client_index in client_indexes:
            training_num += self.train_data_local_num_dict[client_index]

        for client_index in client_indexes:
            average_weight_dict[client_index] = (
                self.train_data_local_num_dict[client_index] / training_num
            )
        return average_weight_dict


    def aggregate(self):
        """
        Aggregate the model updates from clients.

        Returns:
            collections.OrderedDict: The aggregated global model parameters.
        """
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            if len(self.model_dict[idx]) > 0:
                # some workers may not have parameters 
                model_list.append(self.model_dict[idx])
        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))
        # logging.info("################aggregate: %d" % len(model_list))
        # (num0, averaged_params) = model_list[0]
        averaged_params = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_model_params = model_list[i]
                # w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k]
                else:
                    averaged_params[k] += local_model_params[k]


        # server optimizer
        # save optimizer state
        self.opt.zero_grad()
        opt_state = self.opt.state_dict()
        # set new aggregated grad
        self.set_model_global_grads(averaged_params)
        self.opt = self._instantiate_opt()
        # load optimizer state
        self.opt.load_state_dict(opt_state)
        self.opt.step()

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return self.get_global_model_params()

    def set_model_global_grads(self, new_state):
        """
        Set the global model gradients based on a provided dictionary.

        Args:
            new_state (dict): A dictionary containing the new global model gradients.
        """
        new_model = copy.deepcopy(self.aggregator.model)
        new_model.load_state_dict(new_state)
        with torch.no_grad():
            for parameter, new_parameter in zip(self.aggregator.model.parameters(), new_model.parameters()):
                parameter.grad = parameter.data - new_parameter.data
                # because we go to the opposite direction of the gradient
        model_state_dict = self.aggregator.model.state_dict()
        new_model_state_dict = new_model.state_dict()
        for k in dict(self.aggregator.model.named_parameters()).keys():
            new_model_state_dict[k] = model_state_dict[k]
        # self.trainer.model.load_state_dict(new_model_state_dict)
        self.set_global_model_params(new_model_state_dict)

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        """Randomly sample a subset of clients for a communication round.

        Args:
            round_idx (int): The current communication round index.
            client_num_in_total (int): The total number of clients.
            client_num_per_round (int): The number of clients to sample for the round.

        Returns:
            list: A list of indices representing the selected clients for the round.
        """
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        """Generate a subset of the testing dataset for validation.

        Args:
            num_samples (int): The number of samples to include in the validation set.

        Returns:
            torch.utils.data.DataLoader: A DataLoader containing the validation subset.
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
        """Test the global model on all clients.

        Args:
            round_idx (int): The current communication round index.
        """
        if (
            round_idx % self.args.frequency_of_the_test == 0
            or round_idx == self.args.comm_round - 1
        ):
            self.args.round_idx = round_idx
            if round_idx == self.args.comm_round - 1:
                metrics = self.aggregator.test(self.test_global, self.device, self.args)
            else:
                metrics = self.aggregator.test(self.val_global, self.device, self.args)

