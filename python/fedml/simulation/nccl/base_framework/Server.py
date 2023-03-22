import copy
import logging

import numpy as np
import torch
import wandb

from .common import ReduceOp
from .common import broadcast_model_state
from .common import new_group
from .params import LocalAggregatorToServerParams
from .params import ServerToClientParams


class BaseServer:
    """
    Used to manage and aggregate results from local aggregators.
    We hope users does not need to modify this code.
    """

    # def __init__(self, args, trainer, device, dataset, comm=None, rank=0, size=0, backend="NCCL"):
    def __init__(self, args, rank, worker_number, comm, device, dataset, model, trainer):
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
        self.worker_number = worker_number
        self.device_number = worker_number - 1
        self.groups = {}
        for device in range(self.device_number):
            global_rank = device + 1
            self.groups[global_rank] = new_group(ranks=[0, global_rank])
        # self.backend = backend
        logging.info("self.trainer = {}".format(self.trainer))
        self.client_runtime_history = {}

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def simulate_all_tasks(self, server_params, client_indexes):
        localAggregatorToServerParams = LocalAggregatorToServerParams(None)
        # model_update = [torch.zeros_like(v) for v in get_weights(self.trainer.get_model_params())]
        # localAggregatorToServerParams.add_reduce_param(name="model_params",
        #         param=model_update, op=ReduceOp.SUM)
        global_model_params = self.trainer.get_model_params()
        for name, param in global_model_params.items():
            # logging.info(f"name: {name}, param.shape: {param.shape}")
            localAggregatorToServerParams.add_reduce_param(name=name, param=torch.zeros_like(param), op=ReduceOp.SUM)
            # logging.info(f"localAggregatorToServerParams.get({name}): {localAggregatorToServerParams.get(name)}")

        for client_index in client_indexes:
            localAggregatorToServerParams.add_gather_params(client_index, "runtime", torch.tensor(0.0))
        return localAggregatorToServerParams

    def workload_estimate(self, client_indexes, mode="simulate"):
        if mode == "simulate":
            client_samples = [self.train_data_local_num_dict[client_index] for client_index in client_indexes]
            workload = client_samples
        elif mode == "real":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return workload

    def memory_estimate(self, client_indexes, mode="simulate"):
        if mode == "simulate":
            memory = np.ones(self.device_number)
        elif mode == "real":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return memory

    def resource_estimate(self, mode="simulate"):
        if mode == "simulate":
            resource = np.ones(self.device_number)
        elif mode == "real":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return resource

    def client_schedule(self, round_idx, client_num_in_total, client_num_per_round, server_params, mode="simulate"):
        # scheduler(workloads, constraints, memory)
        client_indexes = self.client_sampling(round_idx, client_num_in_total, client_num_per_round)
        # workload = self.workload_estimate(client_indexes, mode)
        # resources = self.resource_estimate(mode)
        # memory = self.memory_estimate(mode)

        # mode = 0
        # my_scheduler = scheduler(workload, resources, memory)
        # schedules = my_scheduler.DP_schedule(mode)
        # for i in range(len(schedules)):
        #     print("Resource %2d: %s\n" % (i, str(schedules[i])))

        client_schedule = np.array_split(client_indexes, self.device_number)
        for i in range(self.device_number):
            simulate_client_indexes = np.zeros([client_num_per_round], dtype=int) - 1
            simulate_client_indexes[: len(client_schedule[i])] = client_schedule[i]
            server_params.add_broadcast_param(name=f"client_schedule{i}", param=torch.tensor(simulate_client_indexes))
        return client_indexes, client_schedule

    def get_average_weight(self, client_indexes):
        average_weight_dict = {}
        training_num = 0
        for client_index in client_indexes:
            training_num += self.train_data_local_num_dict[client_index]

        for client_index in client_indexes:
            average_weight_dict[client_index] = self.train_data_local_num_dict[client_index] / training_num
        return average_weight_dict

    def encode_average_weight_dict(self, server_params, average_weight_dict):
        server_params.add_broadcast_param(
            name="average_weight_dict_keys", param=torch.tensor(list(average_weight_dict.keys()))
        )
        server_params.add_broadcast_param(
            name="average_weight_dict_values", param=torch.tensor(list(average_weight_dict.values()))
        )

    def decode_average_weight_dict(self, server_params):
        pass

    def record_client_runtime(self, client_runtimes):
        pass

    def train(self):
        server_params = ServerToClientParams()
        server_params.add_broadcast_param(name="broadcastTest", param=torch.tensor([1, 2, 3]))
        server_params.broadcast()
        logging.info(f'server_params.get("broadcastTest") {server_params.get("broadcastTest")}')

        for round in range(self.args.comm_round):
            # logging.info(f"self.trainer.model.conv1.weight[0,0,:,:]: \
            #     {self.trainer.model.conv1.weight[0,0,:,:]}")
            broadcast_model_state(self.trainer.model.state_dict(), src=0)
            server_params = ServerToClientParams()
            client_indexes, client_schedule = self.client_schedule(
                round, self.args.client_num_in_total, self.args.client_num_per_round, server_params
            )
            average_weight_dict = self.get_average_weight(client_indexes)
            self.encode_average_weight_dict(server_params, average_weight_dict)
            # model_params = get_weights(self.trainer.get_model_params())
            # server_params.add_broadcast_param(name="model_params", param=model_params)
            server_params.broadcast()
            localAggregatorToServerParams = self.simulate_all_tasks(server_params, client_indexes)
            # logging.info(f"Client Runtime: {localAggregatorToServerParams.get('runtime')}")
            # logging.info(f"localAggregatorToServerParams.get('fc.bias')[:5]: {localAggregatorToServerParams.get('fc.bias')[:5]}, ")
            localAggregatorToServerParams.communicate(self.rank, self.groups, client_schedule)
            # for device_rank in range(self.device_number):
            # localAggregatorToServerParams.add_gather_params(client_index, "runtime", client_runtime)
            client_runtimes = localAggregatorToServerParams.get("runtime")
            logging.info(f"Client Runtime: {client_runtimes}")
            self.record_client_runtime(client_runtimes)
            # logging.info(f"localAggregatorToServerParams.get('fc.bias')[:5]: {localAggregatorToServerParams.get('fc.bias')[:5]}, ")
            # global_model_params = localAggregatorToServerParams.get("model_params")
            # self.trainer.set_model_params(global_model_params)
            # set_model_params_with_list(self.trainer.model, global_model_params)
            # for name, param in self.trainer.model.state_dict().items():
            #     logging.info(f"name:{name}, param.shape: {param.shape}")
            # for name, param in localAggregatorToServerParams.__dict__.items():
            #     logging.info(f"name:{name}, param:{param}")

            global_model_state = self.trainer.model.state_dict()
            for name, param in global_model_state.items():
                param.data = localAggregatorToServerParams.get(name)
            self.trainer.set_model_params(global_model_state)
            # logging.info(f"self.trainer.model.conv1.weight[0,0,:,:]: \
            #     {self.trainer.model.conv1.weight[0,0,:,:]}")
            self.test_on_server_for_all_clients(round)

    def test_on_server_for_all_clients(self, round_idx):
        if self.trainer.test_on_the_server(
            self.train_data_local_dict, self.test_data_local_dict, self.device, self.args,
        ):
            return

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

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                # metrics = self.trainer.test(self.val_global, self.device, self.args)
                metrics = self.trainer.test(self.test_global, self.device, self.args)

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
