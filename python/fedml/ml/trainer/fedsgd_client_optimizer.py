import torch
import logging
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod

from ...core.common.ml_engine_backend import MLEngineBackend
from .base_client_optimizer import ClientOptimizer


from fedml.ml.ml_message import MLMessage

from fedml.utils.model_utils import get_all_bn_params, get_named_data

from fedml.core.compression import MLcompression


class FedSGDClientOptimizer(ClientOptimizer):
    def load_status(self, args, client_status):
        """
        Load status of client optimizer.
        """
        self.client_status = client_status
        if MLcompression.check_args_compress(args) and self.compressor.stateful:
            assert args.local_cache
            self.compressor.load_status(args, self.client_status)


    def add_status(self, client_status):
        if MLcompression.check_args_compress(self.args) and self.compressor.stateful:
            assert self.args.local_cache
            self.compressor.add_status(client_status)
        return client_status


    def preprocess(self, args, client_index, model, train_data, device, server_result, criterion):
        # Not need to create optimizer. This may consume lot of time, as the local iter = 1.
        # if args.client_optimizer == "sgd":
        #     self.optimizer = torch.optim.SGD(
        #         filter(lambda p: p.requires_grad, model.parameters()),
        #         lr=args.learning_rate,
        #         weight_decay=args.weight_decay,
        #     )
        # else:
        #     self.optimizer = torch.optim.Adam(
        #         filter(lambda p: p.requires_grad, model.parameters()),
        #         lr=args.learning_rate,
        #         weight_decay=args.weight_decay,
        #         amsgrad=True,
        #     )
        return model


    def backward(self, args, client_index, model, x, labels, criterion, device, loss):
        """
        """
        loss.backward()
        return loss


    def update(self, args, client_index, model, x, labels, criterion, device):
        """
            SGD return grad to the server, not update at client side
        """
        pass


    def end_local_training(self, args, client_index, model, train_data, device):
        other_result = dict()
        named_grads = get_named_data(model, mode='GRAD', use_cuda=False)
        # if MLcompression.check_args_compress(args):
        #     compressed_named_parameters, params_indexes = \
        #         self.compressor.compress_named_parameters(named_grads, self.args)
        #     other_result[MLMessage.MODEL_PARAMS] = compressed_named_parameters
        #     other_result[MLMessage.MODEL_INDEXES] = params_indexes
        # else:
        #     other_result[MLMessage.MODEL_PARAMS] = named_grads

        bn_params = get_all_bn_params(model)
        other_result["bn_params"] = bn_params
        return other_result

        # return model.cpu().state_dict(), {}














