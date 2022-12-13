import logging
from typing import List, Tuple, Dict
import torch
import copy

from ...core.common.ml_engine_backend import MLEngineBackend

from .agg_operator import FedMLAggOperator
from .base_server_optimizer import ServerOptimizer

from fedml.ml.utils.optrepo import OptRepo
from fedml.ml.utils.opt_utils import OptimizerLoader

from fedml.ml.ml_message import MLMessage


class FedOptServerOptimizer(ServerOptimizer):
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
        self.opt = OptRepo.name2cls(self.args.server_optimizer)(
            model.parameters(),
            lr=self.args.server_lr,
            momentum=self.args.server_momentum, # for fedavgm
            weight_decay=self.args.weight_decay,
            # eps = 1e-3 for adaptive optimizer
        )
        self.model = model
        self.opt.zero_grad()
        self.opt_loader = OptimizerLoader(model, self.opt)


    def get_init_params(self) -> Dict:
        other_result = dict()
        return other_result


    def global_agg_seq(self, args, client_result):
        """ Used in hiearchical and sequentially aggregation. """
        key_op_weight_list = [(MLMessage.MODEL_PARAMS, "weighted_avg", client_result[MLMessage.MODEL_PARAMS]["agg_weight"])]
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
            w_global = copy.deepcopy(agg_params_dict[MLMessage.MODEL_PARAMS]["agg_params"])
        else:
            w_global = FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list)
            # agg_params_dict = self.sync_agg_params(self.client_result_dict, self.sample_num_dict, key_op_list)

        self.opt_loader.set_model_global_grads(w_global)    # set model pseudo grad = w_global - w_current 
        _ = self.opt_loader.update_opt_state(update_model=True)
        self.opt_loader.zero_grad()
        w_global = self.model.cpu().state_dict()
        return w_global


    def before_agg(self, client_result_dict, sample_num_dict):
        self.client_result_dict = client_result_dict
        self.sample_num_dict = sample_num_dict


    def end_agg(self):
        self.initialize_params_dict()
        other_result = dict()
        return other_result





