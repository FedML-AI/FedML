import logging
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

import numpy as np
import torch
import wandb

from ...ml.aggregator.server_optimizer_creator import create_server_optimizer
from ...ml.ml_message import MLMessage
from ...core.contribution.contribution_assessor_manager import ContributionAssessorManager
from ...core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
from ...core.security.fedml_attacker import FedMLAttacker
from ...core.security.fedml_defender import FedMLDefender

from ...core.alg_frame.server_aggregator import ServerAggregator


from fedml.utils.model_utils import transform_tensor_to_list, transform_list_to_tensor

from ...core.schedule.seq_train_scheduler import SeqTrainScheduler
from ...core.schedule.runtime_estimate import t_sample_fit


class HierarchicalGlobalAggregator(ServerAggregator):
    """Abstract base class for federated learning trainer."""

    def __init__(self, model, args, device, worker_num=None):
        super().__init__(model, args)
        self.cpu_transfer = False if not hasattr(self.args, "cpu_transfer") else self.args.cpu_transfer

        self.server_optimizer = create_server_optimizer(args)
        self.server_optimizer.initialize(args, model)

        if worker_num is None:
            self.worker_num = args.client_num_per_round
        else:
            self.worker_num = worker_num

        self.client_num_in_total = self.args.client_num_in_total

        self.device = device
        self.reset_client_result_dict()

        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        self.client_indexes_current_round = None
        self.runtime_history = {}
        self.runtime_avg = {}
        for i in range(self.worker_num):
            self.runtime_history[i] = {}
            self.runtime_avg[i] = {}
            for j in range(self.args.client_num_in_total):
                self.runtime_history[i][j] = []
                self.runtime_avg[i][j] = None


    def reset_client_result_dict(self):
        self.model_dict = dict()            # max length: client_num_in_total
        self.sample_num_dict = dict()       # max length: client_num_in_total
        self.client_result_dict = dict()    # max length: client_num_in_total
        self.worker_result_dict = dict()    # max length: worker_num


    def set_id(self, aggregator_id):
        self.id = aggregator_id

    def get_init_server_result(self):
        server_result = {}
        global_model_params = self.get_model_params()
        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)
        server_result[MLMessage.MODEL_PARAMS] = global_model_params
        # server_result[MLMessage.PARAMS_TO_CLIENT_OPTIMIZER] = self.server_optimizer.get_init_params()
        other_result = self.server_optimizer.get_init_params()
        server_result.update(other_result)
        # logging.info(f"server_result: {server_result}")
        return server_result

    def get_model_params(self):
        if self.cpu_transfer:
            return self.model.cpu().state_dict()
        return self.model.state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)


    def check_whether_all_receive(self):
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True


    def workload_estimate(self, client_indexes, mode="simulate"):
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
        if mode == "simulate":
            memory = np.ones(self.worker_num)
        elif mode == "real":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return memory

    def resource_estimate(self, mode="simulate"):
        if mode == "simulate":
            resource = np.ones(self.worker_num)
        elif mode == "real":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return resource

    def record_client_runtime(self, worker_id, client_runtimes):
        for client_id, runtime in client_runtimes.items():
            self.runtime_history[worker_id][client_id].append(runtime)
        if self.args.runtime_est_mode == 'EMA':
            for client_id, runtime in client_runtimes.items():
                if self.runtime_avg[worker_id][client_id] is None:
                    self.runtime_avg[worker_id][client_id] = runtime
                else:
                    self.runtime_avg[worker_id][client_id] += self.runtime_avg[worker_id][client_id]/2 + runtime/2
        elif self.args.runtime_est_mode == 'time_window':
            for client_id, runtime in client_runtimes.items():
                self.runtime_history[worker_id][client_id] = self.runtime_history[worker_id][client_id][-3:]
        elif self.args.runtime_est_mode == 'history':
            pass
        else:
            raise NotImplementedError

    def generate_client_schedule(self, round_idx, client_indexes):
        # self.runtime_history = {}
        # for i in range(self.worker_num):
        #     self.runtime_history[i] = {}
        #     for j in range(self.args.client_num_in_total):
        #         self.runtime_history[i][j] = []
        things_to_wandb = {}
        previous_time = time.time()
        if hasattr(self.args, "simulation_schedule")  and self.args.simulation_schedule is not None \
            and round_idx > 5:
            # and round_idx > 5:
            # Need some rounds to record some information. 
            simulation_schedule = self.args.simulation_schedule
            if self.args.runtime_est_mode == 'EMA':
                runtime_to_fit = self.runtime_avg
            elif self.args.runtime_est_mode == 'time_window':
                runtime_to_fit = self.runtime_history
            elif self.args.runtime_est_mode == 'history':
                runtime_to_fit = self.runtime_history
            else:
                raise NotImplementedError

            fit_params, fit_funcs, fit_errors = t_sample_fit(
                self.worker_num, self.args.client_num_in_total, runtime_to_fit, 
                self.train_data_local_num_dict, uniform_client=True, uniform_gpu=False)

            if self.args.enable_wandb:
                # wandb.log({"Time_Fit_workload": time.time() - previous_time, "round": round_idx})
                things_to_wandb["Time_Fit_workload"] = time.time() - previous_time
                current_time = time.time()

            logging.info(f"fit_params: {fit_params}")
            if hasattr(self.args, "tracking_runtime") and self.args.tracking_runtime and self.args.enable_wandb:
                for worker_id, fit_param in fit_params.items():
                    things_to_wandb[f"Runtime_fit_param_w{worker_id}_k"] = fit_param[0][0]
                    things_to_wandb[f"Runtime_fit_param_w{worker_id}_b"] = fit_param[0][1]

            logging.info(f"fit_errors: {fit_errors}")
            avg_fit_error = 0.0
            sum_times = 0
            for gpu, gpu_erros in fit_errors.items():
                for client, client_error in gpu_erros.items():
                    avg_fit_error += client_error
                    sum_times += 1
            avg_fit_error /= sum_times
            if self.args.enable_wandb:
                # wandb.log({"RunTimeEstimateError": avg_fit_error, "round": round_idx})
                things_to_wandb["RunTimeEstimateError"] = avg_fit_error

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

            if self.args.enable_wandb:
                # wandb.log({"Time_Schedule": time.time() - current_time, "round": round_idx})
                things_to_wandb["Time_Schedule"] = time.time() - current_time
                current_time = time.time()
        else:
            client_schedule = np.array_split(client_indexes, self.worker_num)
        if self.args.enable_wandb:
            # wandb.log({"RunTimeSchedule": time.time() - previous_time, "round": round_idx})
            things_to_wandb["RunTimeSchedule"] = time.time() - previous_time

        if self.args.enable_wandb:
            things_to_wandb["round"] = round_idx
            wandb.log(things_to_wandb)
        logging.info(f"Schedules: {client_schedule}")
        return client_schedule


    def global_aggregate_seq(self, worker_index, local_agg_client_result, local_sample_num_dict):
        # logging.info("recevice worker result index = %d" % worker_index)
        start_time = time.time()
        if MLMessage.LOCAL_COLLECT_RESULT in local_agg_client_result:
            # local_collect_result: parameters that cannot be locally aggregated.
            local_collect_result = local_agg_client_result[MLMessage.LOCAL_COLLECT_RESULT]
            for client_index, client_result in local_collect_result.items():
                self.sample_num_dict[int(client_index)] = local_sample_num_dict[int(client_index)]
                self.client_result_dict[int(client_index)] = client_result
        if MLMessage.LOCAL_AGG_RESULT in local_agg_client_result:
            # local_collect_result: parameters that are locally aggregated.
            local_agg_result = local_agg_client_result[MLMessage.LOCAL_AGG_RESULT]
            self.server_optimizer.global_agg_seq(self.args, local_agg_result)
        self.flag_client_model_uploaded_dict[worker_index] = True
        end_time = time.time()
        # logging.info("aggregate time cost: %d" % (end_time - start_time))


    # def add_local_trained_result(self, index, client_result, sample_num):
    #     logging.info("add_model. index = %d" % index)
    #     self.model_dict[index] = client_result[MLMessage.MODEL_PARAMS]
    #     self.client_result_dict[index] = client_result
    #     self.sample_num_dict[index] = sample_num
    #     self.flag_client_model_uploaded_dict[index] = True


    def on_before_aggregation(
        self, raw_client_model_or_grad_list: List[Tuple[float, Dict]]
    ) -> List[Tuple[float, Dict]]:
        if FedMLAttacker.get_instance().is_model_attack():
            raw_client_model_or_grad_list = FedMLAttacker.get_instance().attack_model(
                raw_client_grad_list=raw_client_model_or_grad_list,
                extra_auxiliary_info=None,
            )
        if FedMLDefender.get_instance().is_defense_enabled():
            raw_client_model_or_grad_list = FedMLDefender.get_instance().defend_before_aggregation(
                raw_client_grad_list=raw_client_model_or_grad_list,
                extra_auxiliary_info=self.get_model_params(),
            )

        return raw_client_model_or_grad_list



    def get_training_num_in_round(self, client_indexes):
        """
            Used to preload the average weight.
        """
        training_num_in_round = 0
        for client_index in client_indexes:
            training_num_in_round += self.train_data_local_num_dict[client_index]
        return training_num_in_round


    def get_average_weight(self, client_indexes):
        """
            Used to preload the average weight.
        """
        average_weight_dict = {}
        training_num = 0
        for client_index in client_indexes:
            training_num += self.train_data_local_num_dict[client_index]

        for client_index in client_indexes:
            average_weight_dict[client_index] = (
                self.train_data_local_num_dict[client_index] / training_num
            )
        return training_num, average_weight_dict


    def aggregate(self) -> Dict:
        start_time = time.time()

        raw_client_model_or_grad_list = []
        # for idx in range(self.worker_num):
        #     raw_client_model_or_grad_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
        # raw_client_model_or_grad_list = self.on_before_aggregation(raw_client_model_or_grad_list)

        self.server_optimizer.before_agg(self.client_result_dict, self.sample_num_dict)
        if FedMLDefender.get_instance().is_defense_enabled():
            new_global_params = FedMLDefender.get_instance().defend_on_aggregation(
                raw_client_grad_list=raw_client_model_or_grad_list,
                base_aggregation_func=self.server_optimizer.agg,
                extra_auxiliary_info=self.get_model_params(),
            )
        else:
            new_global_params = self.server_optimizer.agg(self.args, raw_client_model_or_grad_list)

        new_global_params = self.on_after_aggregation(new_global_params)

        self.set_model_params(new_global_params)
        # params_to_client_optimizer = self.server_optimizer.end_agg()
        server_result = {}
        if self.args.is_mobile == 1:
            new_global_params = transform_tensor_to_list(new_global_params)
        server_result[MLMessage.MODEL_PARAMS] = new_global_params
        other_result = self.server_optimizer.end_agg()
        server_result.update(other_result)
        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))

        return server_result



    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.client_indexes_current_round = client_indexes
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    # def aggregate(self, raw_client_model_or_grad_list: List[Tuple[float, Dict]]) -> Dict:
    #     if FedMLDefender.get_instance().is_defense_enabled():
    #         return FedMLDefender.get_instance().defend_on_aggregation(
    #             raw_client_grad_list=raw_client_model_or_grad_list,
    #             base_aggregation_func=FedMLAggOperator.agg,
    #             extra_auxiliary_info=self.get_model_params(),
    #         )
    #     return FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list)


    def on_after_aggregation(self, aggregated_model_or_grad: Dict) -> Dict:
        if FedMLDifferentialPrivacy.get_instance().is_global_dp_enabled():
            logging.info("-----add central DP noise ----")
            aggregated_model_or_grad = FedMLDifferentialPrivacy.get_instance().add_global_noise(
                aggregated_model_or_grad
            )
        if FedMLDefender.get_instance().is_defense_enabled():
            aggregated_model_or_grad = FedMLDefender.get_instance().defend_after_aggregation(aggregated_model_or_grad)

        return aggregated_model_or_grad

    def test(self, test_data, device, args):
        pass

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        pass

    def distributed_test(self):
        pass




