import logging
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

import numpy as np
import torch
import wandb

from .server_optimizer_creator import create_server_optimizer
from ..ml_message import MLMessage
from ...core.contribution.contribution_assessor_manager import ContributionAssessorManager
from ...core.dp.fedml_differential_privacy import FedMLDifferentialPrivacy
from ...core.security.fedml_attacker import FedMLAttacker
from ...core.security.fedml_defender import FedMLDefender

from ...core.alg_frame.server_aggregator import ServerAggregator


from fedml.utils.model_utils import transform_tensor_to_list, transform_list_to_tensor

from ...core.schedule.seq_train_scheduler import SeqTrainScheduler
from ...core.schedule.runtime_estimate import t_sample_fit


class HierarchicalLocalAggregator(object):
    """Abstract base class for federated learning trainer."""
    # def __init__(self, model, args, device):
    def __init__(self, args, device):
        # self.model = model
        self.id = 0
        self.args = args
        self.cpu_transfer = False if not hasattr(self.args, "cpu_transfer") else self.args.cpu_transfer

        """Do not need to initialize optimizer with model, only need to call local agg functions"""
        # self.server_optimizer.initialize(args, model)

        self.server_optimizer = create_server_optimizer(self.args)
        self.device = device
        self.reset()

    def reset(self):
        self.client_result_dict = dict()

    def set_id(self, aggregator_id):
        self.id = aggregator_id


    def local_aggregate_seq(self, index, client_result, sample_num, training_num_in_round):
        key_op_weight_list = self.server_optimizer.agg_seq(self.args, index, client_result, sample_num, training_num_in_round)
        for key, op, weight in key_op_weight_list:
            """Delete params that are locally aggregated"""
            # client_result[key] = None
            client_result.pop(key)
        self.client_result_dict[index] = client_result



    def end_local_aggregate_seq(self):
        local_agg_client_result = {}
        local_agg_params_dict = self.server_optimizer.end_agg_seq(self.args)
        # local_client_result.update(local_agg_params_dict)
        local_agg_client_result[MLMessage.LOCAL_AGG_RESULT] = local_agg_params_dict
        local_agg_client_result[MLMessage.LOCAL_COLLECT_RESULT] = self.client_result_dict
        return local_agg_client_result
        # return agg_params_dict






