import logging
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

from ...ml.aggregator.server_optimizer_creator import create_server_optimizer
from ...ml.ml_message import MLMessage
from ...core.contribution.contribution_assessor_manager import ContributionAssessorManager
from ...core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
from ...core.security.fedml_attacker import FedMLAttacker
from ...core.security.fedml_defender import FedMLDefender

from ...core.alg_frame.server_aggregator import ServerAggregator


# from .utils import transform_tensor_to_list, transform_list_to_tensor


class BaseServerAggregator(ServerAggregator):
    """Abstract base class for federated learning trainer."""

    def __init__(self, model, args, device, worker_num=None):
        super().__init__(model, args)
        self.cpu_transfer = False if not hasattr(self.args, "cpu_transfer") else self.args.cpu_transfer

        self.server_optimizer = create_server_optimizer(args)
        self.server_optimizer.initialize(args, model)

        # self.worker_num = args.client_num_in_total
        if worker_num is None:
            self.worker_num = args.client_num_per_round
        else:
            self.worker_num = worker_num

        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False


    def set_id(self, aggregator_id):
        self.id = aggregator_id

    def get_init_server_result(self):
        server_result = {}
        global_model_params = self.get_model_params()
        # if self.args.is_mobile == 1:
        #     global_model_params = transform_tensor_to_list(global_model_params)

        server_result[MLMessage.MODEL_PARAMS] = global_model_params
        server_result[MLMessage.PARAMS_TO_CLIENT_OPTIMIZER] = self.server_optimizer.get_init_params()
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


    def add_local_trained_result(self, index, client_result, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = client_result[MLMessage.MODEL_PARAMS]

        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True
        self.__add_params_to_server_optimizer(index, client_result[MLMessage.PARAMS_TO_SERVER_OPTIMIZER])


    def add_local_trained_result_inplace(self, index, client_result, sample_num):
        pass


    def __add_params_to_server_optimizer(self, index, params_to_server_optimizer):
        self.server_optimizer.add_params_to_server_optimizer(index, params_to_server_optimizer)


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

    def aggregate(self) -> Dict:
        start_time = time.time()

        raw_client_model_or_grad_list = []
        for idx in range(self.worker_num):
            raw_client_model_or_grad_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
        raw_client_model_or_grad_list = self.on_before_aggregation(raw_client_model_or_grad_list)

        self.server_optimizer.before_agg()
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
        params_to_client_optimizer = self.server_optimizer.end_agg()

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))

        server_result = {}
        # if self.args.is_mobile == 1:
        #     new_global_params = transform_tensor_to_list(new_global_params)
        server_result[MLMessage.MODEL_PARAMS] = new_global_params
        server_result[MLMessage.PARAMS_TO_CLIENT_OPTIMIZER] = params_to_client_optimizer

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




