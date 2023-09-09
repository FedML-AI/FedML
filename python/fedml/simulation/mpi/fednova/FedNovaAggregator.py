import logging
import logging
import random
import time

import numpy as np
import torch
import wandb

from .utils import transform_list_to_tensor
from ....core.schedule.runtime_estimate import t_sample_fit
from ....core.schedule.seq_train_scheduler import SeqTrainScheduler


class FedNovaAggregator(object):
    """
    Federated Nova Aggregator for aggregating local model updates in a federated learning setup.

    This class manages the aggregation of local model updates from multiple clients in a federated learning system
    using the Federated Nova (FedNova) approach.

    Args:
        train_global (Dataset): Global training dataset.
        test_global (Dataset): Global test/validation dataset.
        all_train_data_num (int): Total number of training samples across all clients.
        train_data_local_dict (dict): Dictionary containing local training datasets for each client.
        test_data_local_dict (dict): Dictionary containing local test/validation datasets for each client.
        train_data_local_num_dict (dict): Dictionary containing the number of local training samples for each client.
        worker_num (int): Number of worker nodes/clients.
        device: The computing device (e.g., GPU) to perform aggregation and computations.
        args: Additional configuration and hyperparameters.
        server_aggregator: Server-side aggregator for aggregation methods.

    Methods:
        get_global_model_params():
            Get the global model parameters.

        set_global_model_params(model_parameters):
            Set the global model parameters.

        add_local_trained_result(index, local_result):
            Add the local training results from a client.

        check_whether_all_receive():
            Check if all clients have uploaded their local models.

        record_client_runtime(worker_id, client_runtimes):
            Record client runtime information for scheduling.

        generate_client_schedule(round_idx, client_indexes):
            Generate a schedule for client training in the federated round.

        get_average_weight(client_indexes):
            Calculate the average weight for client selection.

        fednova_aggregate(params, norm_grads, tau_effs, tau_eff=0):
            Perform FedNova aggregation of local model updates.

        aggregate():
            Aggregate local model updates using the FedNova aggregation method.

        client_sampling(round_idx, client_num_in_total, client_num_per_round):
            Perform client sampling for a federated round.

        _generate_validation_set(num_samples=10000):
            Generate a validation dataset for testing.

        test_on_server_for_all_clients(round_idx):
            Test the global model on all clients' datasets.

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
        """
        Initialize the FedNova manager.

        Args:
            train_global: Global training dataset.
            test_global: Global test dataset.
            all_train_data_num: Total number of training samples.
            train_data_local_dict: Dictionary containing local training datasets for clients.
            test_data_local_dict: Dictionary containing local test datasets for clients.
            train_data_local_num_dict: Dictionary containing the number of local training samples for clients.
            worker_num: Number of worker nodes (clients).
            device: PyTorch device (CPU or GPU) to run computations.
            args: Command-line arguments.
            server_aggregator: Aggregator for model updates.
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
        self.result_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        self.runtime_history = {}
        for i in range(self.worker_num):
            self.runtime_history[i] = {}
            for j in range(self.args.client_num_in_total):
                self.runtime_history[i][j] = []

        # ====================================================
        self.global_momentum_buffer = dict()

    def get_global_model_params(self):
        """
        Get the global model parameters.

        Returns:
            dict: Global model parameters.
        """
        return self.aggregator.get_model_params()

    def set_global_model_params(self, model_parameters):
        """
        Set the global model parameters.

        Args:
            model_parameters (dict): Global model parameters to be set.
        """
        self.aggregator.set_model_params(model_parameters)

    def add_local_trained_result(self, index, local_result):
        """
        Add the local training result for a client.

        Args:
            index (int): Index of the client.
            local_result (dict): Local training result.
        """
        logging.info("add_model. index = %d" % index)
        self.result_dict[index] = local_result
        # self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        """
        Check if all clients have uploaded their local model updates.

        Returns:
            bool: True if all clients have uploaded, False otherwise.
        """
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def record_client_runtime(self, worker_id, client_runtimes):
        """
        Record client runtime information.

        Args:
            worker_id (int): Index of the worker.
            client_runtimes (dict): Dictionary containing client runtime information.
        """
        for client_id, runtime in client_runtimes.items():
            self.runtime_history[worker_id][client_id].append(runtime)

    def generate_client_schedule(self, round_idx, client_indexes):
        """
        Generate a schedule for selecting clients in the current round.

        Args:
            round_idx (int): Current round index.
            client_indexes (list): List of client indexes.

        Returns:
            list: List of client schedules for each worker.
        """
        # self.runtime_history = {}
        # for i in range(self.worker_num):
        #     self.runtime_history[i] = {}
        #     for j in range(self.args.client_num_in_total):
        #         self.runtime_history[i][j] = []

        if hasattr(self.args, "simulation_schedule") and round_idx > 5:
            # Need some rounds to record some information.
            simulation_schedule = self.args.simulation_schedule
            fit_params, fit_funcs, fit_errors = t_sample_fit(
                self.worker_num,
                self.args.client_num_in_total,
                self.runtime_history,
                self.train_data_local_num_dict,
                uniform_client=True,
                uniform_gpu=False,
            )
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
            workloads = np.array([self.train_data_local_num_dict[client_id] for client_id in client_indexes])
            constraints = np.array([1] * self.worker_num)
            memory = np.array([100])
            my_scheduler = SeqTrainScheduler(
                workloads, constraints, memory, fit_funcs, uniform_client=True, uniform_gpu=False
            )
            # my_scheduler = SeqTrainScheduler(workloads, constraints, memory, self.train_data_local_num_dict,
            #     fit_funcs, uniform_client=True, uniform_gpu=False)
            y_schedule, output_schedules = my_scheduler.DP_schedule(mode)
            client_schedule = []
            for indexes in y_schedule:
                client_schedule.append(client_indexes[indexes])
            logging.info(f"Schedules: {client_schedule}")
        else:
            client_schedule = np.array_split(client_indexes, self.worker_num)
        return client_schedule

    def get_average_weight(self, client_indexes):
        """
        Get the average weight for clients based on the number of local samples.

        Args:
            client_indexes (list): List of client indexes.

        Returns:
            dict: Dictionary mapping client index to average weight.
        """
        average_weight_dict = {}
        training_num = 0
        for client_index in client_indexes:
            training_num += self.train_data_local_num_dict[client_index]

        for client_index in client_indexes:
            average_weight_dict[client_index] = self.train_data_local_num_dict[client_index] / training_num
        return average_weight_dict

    def fednova_aggregate(self, params, norm_grads, tau_effs, tau_eff=0):
        """
        Perform FedNova aggregation.

        Args:
            params (dict): Model parameters to be aggregated.
            norm_grads (list): List of normalized gradients from clients.
            tau_effs (list): List of effective tau values.
            tau_eff (int): Effective tau for aggregation (optional).

        Returns:
            dict: Aggregated model parameters.
        """
        # get tau_eff
        if tau_eff == 0:
            tau_eff = sum(tau_effs)
        # get cum grad
        # cum_grad = tau_eff * sum(norm_grads)
        cum_grad = norm_grads[0]
        for k in norm_grads[0].keys():
            for i in range(0, len(norm_grads)):
                if i == 0:
                    cum_grad[k] = norm_grads[i][k] * tau_eff
                else:
                    cum_grad[k] += norm_grads[i][k] * tau_eff
        # update params
        for k in params.keys():
            if self.args.gmf != 0:
                if k not in self.global_momentum_buffer:
                    buf = self.global_momentum_buffer[k] = torch.clone(cum_grad[k]).detach()
                    buf.div_(self.args.learning_rate)
                else:
                    buf = self.global_momentum_buffer[k]
                    buf.mul_(self.args.gmf).add_(1 / self.args.learning_rate, cum_grad[k])
                params[k].sub_(self.args.learning_rate, buf.to(params[k].device))
            else:
                params[k].sub_(cum_grad[k].to(params[k].device))

        return params

    def aggregate(self):
        """
        Aggregate model updates from clients.

        Returns:
            dict: Aggregated model parameters.
        """
        start_time = time.time()
        grad_results = []
        t_eff_results = []
        training_num = 0

        for idx in range(self.worker_num):
            if len(self.result_dict[idx]) > 0:
                # some workers may not have parameters
                # for client_index, client_result in self.result_dict[idx].items():
                for client_result in self.result_dict[idx]:
                    grad_results.append(client_result["grad"])
                    t_eff_results.append(client_result["t_eff"])
            # training_num += self.sample_num_dict[idx]
        logging.info("len of self.result_dict[idx] = " + str(len(self.result_dict)))

        # update the global model which is cached at the server side
        init_params = self.get_global_model_params()
        w_global = self.fednova_aggregate(init_params, grad_results, t_eff_results)
        self.set_global_model_params(w_global)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return w_global

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        """
        Sample clients for the current round.

        Args:
            round_idx (int): Current round index.
            client_num_in_total (int): Total number of clients.
            client_num_per_round (int): Number of clients to sample per round.

        Returns:
            list: List of sampled client indexes.
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
        """
        Generate a validation set for testing.

        Args:
            num_samples (int): Number of samples to include in the validation set (optional).

        Returns:
            DataLoader: DataLoader for the validation set.
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
        """
        Perform testing on the server for all clients.

        Args:
            round_idx (int): Current round index.
        """
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))
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

            self.args.round_idx = round_idx
            if round_idx == self.args.comm_round - 1:
                metrics = self.aggregator.test(self.test_global, self.device, self.args)
            else:
                metrics = self.aggregator.test(self.val_global, self.device, self.args)
