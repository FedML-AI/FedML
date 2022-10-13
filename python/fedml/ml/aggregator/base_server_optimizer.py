import logging
import copy
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod

from ...core.common.ml_engine_backend import MLEngineBackend

from .agg_operator import FedMLAggOperator



class ServerOptimizer(ABC):
    """Abstract base class for federated learning trainer.
    1. The goal of this abstract class is to be compatible to
    any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
    2. This class is used only on server side.
    3. This class is an operator which can caches states across rounds. 
    Because the server consistently exists.
    """
    def __init__(self, args, worker_num=None):
        self.args = args
        if worker_num is None:
            self.worker_num = args.client_num_per_round
        else:
            self.worker_num = worker_num
        self.initialize_params_dict()


    def initialize_params_dict(self):
        self.client_index_list = []
        # self.params_to_server_optimizer_dict = dict()
        self.agg_params = None
        self.agg_weight = 0.0
        # save other parmas that need to be seq aggregated. (key: {"agg_weight": agg_weight, "agg_params": agg_params})
        # 
        self.agg_params_dict = {}  



    @abstractmethod
    def initialize(self, args, model):
        pass

    @abstractmethod
    def get_init_params(self) -> Dict:
        """
        1. Return init other_result for special aggregator need.
        """
        pass
        # params_to_client_optimizer = dict()
        # return params_to_client_optimizer
        other_result = dict()
        return other_result


    # def add_params_to_server_optimizer(self, index, params_to_server_optimizer,):
    #     self.client_index_list.append(index)
    #     self.params_to_server_optimizer_dict[index] = params_to_server_optimizer


    def global_seq_agg_params(self, client_result, key_op_weight_list):
        for key, op, weight in key_op_weight_list:
            if key not in self.agg_params_dict:
                self.agg_params_dict[key] = {}
                self.agg_params_dict[key]["agg_weight"] = 0.0
                self.agg_params_dict[key]["agg_params"] = None
            logging.info(f"global seq agg: key: {key}, op: {op}, weight: {weight}")
            self.agg_params_dict[key]["agg_weight"], self.agg_params_dict[key]["agg_params"] = FedMLAggOperator.agg_seq(
                self.args, agg_params=self.agg_params_dict[key]["agg_params"], new_params=client_result[key]["agg_params"],
                agg_weight=self.agg_params_dict[key]["agg_weight"], avg_weight=weight, op=op)


    def seq_agg_params(self, client_result, key_op_weight_list):
        for key, op, weight in key_op_weight_list:
            if key not in self.agg_params_dict:
                self.agg_params_dict[key] = {}
                self.agg_params_dict[key]["agg_weight"] = 0.0
                self.agg_params_dict[key]["agg_params"] = None
            logging.info(f"local seq agg: key: {key}, op: {op}, weight: {weight}")
            self.agg_params_dict[key]["agg_weight"], self.agg_params_dict[key]["agg_params"] = FedMLAggOperator.agg_seq(
                self.args, agg_params=self.agg_params_dict[key]["agg_params"], new_params=client_result[key],
                agg_weight=self.agg_params_dict[key]["agg_weight"], avg_weight=weight, op=op)


    def end_seq_agg_params(self, args, key_op_list):
        for key, op in key_op_list:
            logging.info(f"End seq aggregating, Weight is :  {self.agg_params_dict[key]['agg_weight']}")
            self.agg_params_dict[key]["agg_params"] = FedMLAggOperator.end_agg_seq(
                args, self.agg_params_dict[key]["agg_params"], self.agg_params_dict[key]["agg_weight"], op=op)
        agg_params_dict = copy.deepcopy(self.agg_params_dict)
        """Reset self.agg_params_dict for next round aggregation"""
        self.agg_params_dict = {}
        return agg_params_dict


    def sync_agg_params(self, client_result_dict, sample_num_dict, key_op_list):
        # for i in range(len(sample_num_dict)):
        training_num = 0
        # for client_index in self.client_index_list:
        for client_index in range(self.worker_num):
            local_sample_num = sample_num_dict[client_index]
            training_num += local_sample_num

        for key, op, in key_op_list:
            self.agg_params_dict[key] = {}
            params_list = []
            # for client_index in self.client_index_list:
            #     params_list.append((sample_num_dict[client_index], 
            #         self.params_to_server_optimizer_dict[client_index][key]))
            for client_index in range(self.worker_num):
                params_list.append((sample_num_dict[client_index], 
                    client_result_dict[client_index][key]))

            agg_params = FedMLAggOperator.agg_with_weight(self.args, params_list, training_num, op)
            self.agg_params_dict[key]["agg_params"] = agg_params
        return self.agg_params_dict


    def agg_seq(self, args, index, client_result, sample_num, training_num_in_round):
        """
        Use this function to sequentially add client result
        """
        pass

    def end_agg_seq(self, args):
        """
        Use this function to obtain the final global model when sequentially adding result
        """
        pass



    @abstractmethod
    def agg(self, args, raw_client_model_or_grad_list):
        """
        Use this function to obtain the final global model.
        """
        pass



    def before_agg(self, client_result_dict, sample_num_dict):
        self.client_result_dict = client_result_dict


    def end_agg(self) -> Dict:
        """
        1. Clear self.params_to_server_optimizer_dict 
        2. Return params_to_client_optimizer for special aggregator need.
        """
        self.initialize_params_dict()
        other_result = dict()
        return other_result











