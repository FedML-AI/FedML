import torch
import logging
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
from copy import deepcopy

from ...core.common.ml_engine_backend import MLEngineBackend
from .base_client_optimizer import ClientOptimizer


from fedml.ml.ml_message import MLMessage
from fedml.utils.model_utils import set_model_bn_params
from fedml.utils.model_utils import (
    get_all_bn_params, get_named_data, get_name_params_difference, get_model_name_params_difference,
    add_weights)
from fedml.utils.model_utils import named_params_to


# from fedml.core.compression import MLcompression



class FedDLCClientOptimizer(ClientOptimizer):

    def preprocess(self, args, client_index, model, train_data, device, model_optimizer, criterion):
        self.model_optimizer = model_optimizer
        if args.round_idx == 0 or args.feddlc_download_dense:
            server_weights = self.server_result.get(MLMessage.MODEL_PARAMS)
            model.load_state_dict(server_weights)
        else:
            server_update = named_params_to(self.server_result.get(MLMessage.MODEL_PARAMS), device)
            if hasattr(self.args, "hierarchical_agg") and self.args.hierarchical_agg:
                """
                Use the pseudo grad from the server need to cache the local model.
                Directly cache here for saving memory.
                """
                if "cache_client_model" not in self.server_result:
                    weights = model.state_dict()
                    self.server_result.add("cache_client_model", deepcopy(weights))
                else:
                    weights = deepcopy(self.server_result["cache_client_model"])
            else:
                weights = model.state_dict()

            with torch.no_grad():
                for key in weights.keys():
                    # logging.info(f"param.data.device:{param.data.device}, server_update[name].device: {server_update[name].device}")
                    weights[key] = add_weights(weights[key], server_update[key])
                    # logging.info(f"server_update[name].norm():{server_update[name].norm()}")
            # bn_params = self.server_result.get("bn_params")
            # set_model_bn_params(model, bn_params)
            model.load_state_dict(weights)
            # logging.info(f"client: weights)['fc1'][:3,:3,:3]: {weights['fc1.weight'][:3,:3]}")

        self.prev_model = deepcopy(model)
        return model

    def backward(self, args, client_index, model, x, labels, criterion, device, loss):
        """
        """
        loss.backward()
        return loss


    def update(self, args, client_index, model, x, labels, criterion, device):
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.model_optimizer.step()


    def end_local_training(self, args, client_index, model, train_data, device):
        other_result = dict()
        with torch.no_grad():
            pseudo_grad = get_model_name_params_difference(model, self.prev_model)
        other_result[MLMessage.MODEL_PARAMS] = named_params_to(pseudo_grad, "cpu")
        bn_params = get_all_bn_params(model)
        other_result["bn_params"] = bn_params
        model.load_state_dict(self.prev_model.state_dict())
        return other_result









