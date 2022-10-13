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


class MimeServerOptimizer(ServerOptimizer):
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
        self.grad_global = {
            key: torch.zeros(params.shape)
            for key, params in model.state_dict().items()
        }
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
        self.named_states = self.opt_loader.get_opt_state()


    def get_init_params(self) -> Dict:
        other_result = dict()
        other_result["grad_global"] = self.grad_global
        other_result["global_named_states"] = self.named_states
        return other_result


    def global_agg_seq(self, args, client_result):
        """ Used in hiearchical and sequentially aggregation. """
        key_op_weight_list = [(MLMessage.MODEL_PARAMS, "weighted_avg", client_result[MLMessage.MODEL_PARAMS]["agg_weight"]),
                              ("local_grad", "weighted_avg", client_result[MLMessage.MODEL_PARAMS]["agg_weight"])]
        self.global_seq_agg_params(client_result, key_op_weight_list)
        return key_op_weight_list


    def agg_seq(self, args, index, client_result, sample_num, training_num_in_round):
        """
        Use this function to obtain the final global model.
        """
        key_op_weight_list = [(MLMessage.MODEL_PARAMS, "weighted_avg", sample_num/training_num_in_round),
                              ("local_grad", "weighted_avg", sample_num/training_num_in_round)]
        self.seq_agg_params(client_result, key_op_weight_list)
        return key_op_weight_list


    def end_agg_seq(self, args):
        key_op_list = [(MLMessage.MODEL_PARAMS, "weighted_avg"),
                       ("local_grad", "weighted_avg")]
        agg_params_dict = self.end_seq_agg_params(args, key_op_list)
        return agg_params_dict

    def agg(self, args, raw_client_model_or_grad_list):
        """
        Use this function to obtain the final global model.
        """
        if hasattr(self.args, "aggregate_seq") and self.args.aggregate_seq:
            agg_params_dict = self.end_agg_seq(args)
            w_global = copy.deepcopy(agg_params_dict[MLMessage.MODEL_PARAMS]["agg_params"])
            self.grad_global = copy.deepcopy(agg_params_dict["local_grad"]["agg_params"])
        else:
            w_global = FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list)
            key_op_list = [("local_grad", "weighted_avg")]
            agg_params_dict = self.sync_agg_params(self.client_result_dict, self.sample_num_dict, key_op_list)
            self.grad_global = agg_params_dict["local_grad"]["agg_params"]

        self.opt_loader.set_grad(copy.deepcopy(self.grad_global))
        self.model.load_state_dict(w_global)
        self.named_states = self.opt_loader.update_opt_state(update_model=True)
        w_global = self.model.cpu().state_dict()
        self.opt_loader.zero_grad()

        return w_global


    def before_agg(self, client_result_dict, sample_num_dict):
        self.client_result_dict = client_result_dict
        self.sample_num_dict = sample_num_dict


    def end_agg(self) -> Dict:
        """
        1. Clear self.params_to_server_optimizer_dict 
        2. Return params_to_client_optimizer for special aggregator need.
        """
        self.initialize_params_dict()
        other_result = dict()
        other_result["grad_global"] = self.grad_global
        other_result["global_named_states"] = self.named_states
        return other_result




    






