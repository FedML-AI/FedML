import logging
import time

import numpy as np
import torch

from .common import ReduceOp
from .common import broadcast_model_state
from .common import new_group
from .params import ClientToLocalAggregatorParams
from .params import LocalAggregatorToServerParams
from .params import ServerToClientParams


class BaseLocalAggregator(object):
    """
    Used to manage and aggregate results from local trainers (clients).
    It needs to know all datasets.
    device: indicates the device of this local aggregator
    
    Args:
        args: The command-line arguments for the aggregator.
        rank (int): The rank of this local aggregator.
        worker_number (int): The total number of workers, including the server and clients.
        comm: The communication state.
        device: The device where the aggregator is located.
        dataset: The dataset used for training and testing.
        model: The model used for training.
        trainer: The trainer responsible for training the model.

    Attributes:
        device: Indicates the device of this local aggregator.
        args: The command-line arguments for the aggregator.
        trainer: The trainer responsible for training the model.
        train_global: The global training dataset.
        test_global: The global testing dataset.
        val_global: The global validation dataset (if available).
        train_data_num_in_total: The total number of training data points across all clients.
        test_data_num_in_total: The total number of testing data points across all clients.
        train_data_local_num_dict: A dictionary mapping client indices to the number of training data points for each client.
        train_data_local_dict: A dictionary mapping client indices to their local training datasets.
        test_data_local_dict: A dictionary mapping client indices to their local testing datasets.
        comm: The communication state.
        rank: The rank of this local aggregator.
        device_rank: The rank of this local aggregator as a device (GPU).
        worker_number: The total number of workers, including the server and clients.
        device_number: The total number of devices (GPUs) used for training.
        groups: A dictionary of communication groups, where each group is associated with a specific device.
        
    Methods:
        measure_client_runtime(): Measures the runtime of client operations.
        simulate_client(server_params, client_index, average_weight): Simulates a client's training process.
        add_client_result(localAggregatorToServerParams, client_params): Adds client results to be aggregated and sent to the server.
    """

    # def __init__(self, args, trainer, device, dataset, comm=None, rank=0, size=0, backend="NCCL"):
    def __init__(self, args, rank, worker_number, comm, device, dataset, model, trainer):
        """
        Measure the runtime of client operations.

        This method measures the runtime of client operations and can be used for performance analysis.

        Returns:
            None
        """
        self.device = device
        self.args = args
        self.trainer = trainer
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.comm = comm
        self.rank = rank
        self.device_rank = self.rank - 1
        self.worker_number = worker_number
        self.device_number = worker_number - 1
        # for device in range(self.device_number):
        #     rank = device + 1
        #     if rank == self.rank:
        #         self.group = new_group(ranks=[0, self.rank])
        #     else:
        #         group = new_group(ranks=[0, rank])
        self.groups = {}
        for device in range(self.device_number):
            global_rank = device + 1
            self.groups[global_rank] = new_group(ranks=[0, global_rank])

        # self.backend = backend
        logging.info("self.trainer = {}".format(self.trainer))

    def measure_client_runtime(self):
        """
        Measure the runtime of client operations.

        This method measures the runtime of client operations and can be used for performance analysis.

        Returns:
            None
        """
        pass

    def simulate_client(self, server_params, client_index, average_weight):
        """
        Simulate a client's training process.

        Args:
            server_params: Parameters received from the server.
            client_index (int): The index of the simulated client.
            average_weight: The average weight used in the simulation.

        Returns:
            client_params: Parameters to be sent back to the server.
        """
        
        # server_model_parameters = server_params.get("model_params")
        # self.trainer.set_model_params(server_model_parameters)
        self.trainer.id = client_index
        logging.info(f"Simulating client {client_index}... weight: {average_weight}")

        self.trainer.train(self.train_data_local_dict[client_index], self.device, args=self.args)
        client_params = ClientToLocalAggregatorParams(client_index=client_index)
        client_model_params = self.trainer.get_model_params()
        # logging.info(f"client_model_params.get('fc.bias')[:5]: {client_model_params.get('fc.bias')[:5]}, ")
        for name, param in client_model_params.items():
            client_params.add_reduce_param(name=name, param=param * average_weight, op=ReduceOp.SUM)
        # client_params.add_reduce_param(name="model_params", param=get_weights(client_model_params), op=ReduceOp.SUM)
        # logging.info(f"client_params.get('fc.bias')[:5]: {client_params.get('fc.bias')[:5]}, ")
        return client_params

    def add_client_result(self, localAggregatorToServerParams, client_params):
        """
        Add client results to be aggregated and sent to the server.

        Args:
            localAggregatorToServerParams: Parameters to be sent to the server.
            client_params: Parameters received from a client.

        Returns:
            None
        """
        # Add params that needed to be reduces from clients
        mean_sum_param_names = client_params.get_sum_reduce_param_names()
        for name in mean_sum_param_names:
            localAggregatorToServerParams.add_reduce_param(name=name, param=client_params.get(name), op=ReduceOp.SUM)

        # Add params that needed to be gathered from clients
        gather_param_names = client_params.get_gather_param_names()
        for name in gather_param_names:
            localAggregatorToServerParams.add_gather_params(
                client_index=client_params.client_index, name=name, param=client_params.get(name)
            )

    def simulate_all_tasks(self, server_params):
        """
        Simulate all tasks for this local aggregator.

        Args:
            server_params: Parameters received from the server.

        Returns:
            localAggregatorToServerParams: Parameters to be sent back to the server.
        """
        average_weight_dict = self.decode_average_weight_dict(server_params)
        client_indexes = server_params.get(f"client_schedule{self.device_rank}").numpy()
        simulated_client_indexes = []
        for client_index in client_indexes:
            if client_index < 0:
                continue
            simulated_client_indexes.append(client_index)
        localAggregatorToServerParams = LocalAggregatorToServerParams(simulated_client_indexes)
        logging.info(f"average_weight_dict: {average_weight_dict}")
        # for client_index in client_indexes:
        for client_index in simulated_client_indexes:
            # if client_index < 0:
            #     continue
            start_time = time.time()

            client_params = self.simulate_client(
                server_params, client_index, average_weight=average_weight_dict[client_index]
            )
            self.add_client_result(localAggregatorToServerParams, client_params)
            end_time = time.time()
            client_runtime = torch.tensor(end_time - start_time)
            localAggregatorToServerParams.add_gather_params(client_index, "runtime", client_runtime)

        # logging.info(f"localAggregatorToServerParams.get('fc.bias')[:5]: {localAggregatorToServerParams.get('fc.bias')[:5]}, ")
        return localAggregatorToServerParams

    def client_schedule(self, round_idx, client_num_in_total, client_num_per_round, server_params):
        """
        Receive server's schedule of client indexes for this local aggregator.

        Args:
            round_idx: The current round index.
            client_num_in_total: The total number of clients.
            client_num_per_round: The number of clients to be scheduled for this round.
            server_params: Parameters received from the server.

        Returns:
            None
        """
        # scheduler(workloads, constraints, memory)
        for i in range(self.device_number):
            simulate_client_indexes = np.zeros([client_num_per_round], dtype=int) - 1
            server_params.add_broadcast_param(name=f"client_schedule{i}", param=torch.tensor(simulate_client_indexes))
        return None, None

    def get_average_weight(self, client_indexes):
        """
        Get average weight for a list of client indexes.

        Args:
            client_indexes: A list of client indexes.

        Returns:
            average_weight_dict: A dictionary mapping client indexes to their average weights.
        """
        average_weight_dict = {}
        for client_index in client_indexes:
            average_weight_dict[client_index] = 0.0
        return average_weight_dict

    def encode_average_weight_dict(self, server_params, average_weight_dict):
        """
        Encode and add the average weight dictionary to server parameters.

        Args:
            server_params: Parameters to be sent to the server.
            average_weight_dict: A dictionary mapping client indexes to their average weights.

        Returns:
            None
        """
        server_params.add_broadcast_param(
            name="average_weight_dict_keys", param=torch.tensor(list(average_weight_dict.keys()))
        )
        server_params.add_broadcast_param(
            name="average_weight_dict_values", param=torch.tensor(list(average_weight_dict.values()))
        )

    def decode_average_weight_dict(self, server_params):
        """
        Decode the average weight dictionary from server parameters.

        Args:
            server_params: Parameters received from the server.

        Returns:
            average_weight_dict: A dictionary mapping client indexes to their average weights.
        """
        average_weight_dict_keys = server_params.get("average_weight_dict_keys").numpy()
        average_weight_dict_values = server_params.get("average_weight_dict_values").numpy()
        average_weight_dict = {}
        average_weight_dict = dict(zip(average_weight_dict_keys, average_weight_dict_values))
        return average_weight_dict

    def train(self):
        """
        Train the federated learning model.

        This method handles the federated learning training process, including communication with the server,
        scheduling clients, and aggregating local client results.

        Returns:
            None
        """
        server_params = ServerToClientParams()
        server_params.add_broadcast_param(name="broadcastTest", param=torch.tensor([0, 0, 0]))
        server_params.broadcast()
        logging.info(f'server_params.get("broadcastTest") {server_params.get("broadcastTest")}')
        # for name, param in self.trainer.model.state_dict().items():
        #     logging.info(f"name:{name}, param.shape: {param.shape}")
        for round in range(self.args.comm_round):
            broadcast_model_state(self.trainer.model.state_dict(), src=0)
            server_params = ServerToClientParams()
            _, _ = self.client_schedule(
                round, self.args.client_num_in_total, self.args.client_num_per_round, server_params
            )
            average_weight_dict = self.get_average_weight(list(range(self.args.client_num_per_round)))
            self.encode_average_weight_dict(server_params, average_weight_dict)
            # model_params = get_weights(self.trainer.get_model_params())
            # server_params.add_broadcast_param(name="model_params", param=model_params)
            server_params.broadcast()
            localAggregatorToServerParams = self.simulate_all_tasks(server_params)
            logging.info(f"Client Runtime: {localAggregatorToServerParams.get('runtime')}")
            localAggregatorToServerParams.communicate(self.rank, self.groups)
