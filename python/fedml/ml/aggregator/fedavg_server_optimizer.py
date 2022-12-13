import logging
import copy
from typing import List, Tuple, Dict

from ...core.common.ml_engine_backend import MLEngineBackend

from .agg_operator import FedMLAggOperator
from .base_server_optimizer import ServerOptimizer

from fedml.ml.ml_message import MLMessage

class FedAvgServerOptimizer(ServerOptimizer):
    """Abstract base class for federated learning trainer.
    1. The goal of this abstract class is to be compatible to
    any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
    2. This class is used only on server side.
    3. This class is an operator which can caches states across rounds. 
    Because the server consistently exists.
    """
    def __init__(self, args):
        super().__init__(args)

    def initialize(self, args, model):
        pass


    def get_init_params(self) -> Dict:                                                                    
        other_result = dict()
        return other_result


    def global_agg_seq(self, args, client_result):
        """ Used in hiearchical and sequentially aggregation. """
        key_op_weight_list = [(MLMessage.MODEL_PARAMS, "weighted_avg", client_result[MLMessage.MODEL_PARAMS]["agg_weight"])]
        # logging.info(f"client_result: {list(client_result.keys())}")
        self.global_seq_agg_params(client_result, key_op_weight_list)
        return key_op_weight_list


    def agg_seq(self, args, index, client_result, sample_num, training_num_in_round):
        """
        Use this function to obtain the final global model.
        """
        key_op_weight_list = [(MLMessage.MODEL_PARAMS, "weighted_avg", sample_num/training_num_in_round)]
        self.seq_agg_params(client_result, key_op_weight_list)
        return key_op_weight_list


    def end_agg_seq(self, args):
        key_op_list = [(MLMessage.MODEL_PARAMS, "weighted_avg")]
        agg_params_dict = self.end_seq_agg_params(args, key_op_list)
        return agg_params_dict


    def agg(self, args, raw_client_model_or_grad_list):
        """
        Use this function to obtain the final global model.
        """
        if hasattr(self.args, "aggregate_seq") and self.args.aggregate_seq:
            agg_params_dict = self.end_agg_seq(args)
            new_global_params = copy.deepcopy(agg_params_dict[MLMessage.MODEL_PARAMS]["agg_params"])
            return new_global_params
        else:
            return FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list)


    def before_agg(self, client_result_dict, sample_num_dict):
        self.client_result_dict = client_result_dict


    def end_agg(self) -> Dict:
        self.initialize_params_dict()
        other_result = dict()
        return other_result




    






