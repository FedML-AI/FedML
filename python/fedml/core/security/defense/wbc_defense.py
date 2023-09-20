from collections import OrderedDict
import torch
from typing import Callable, List, Tuple, Dict, Any
import numpy as np
import logging
from .defense_base import BaseDefenseMethod
from ..common import utils

"""
ref: Sun, Jingwei, et al. 
"Fl-wbc: Enhancing robustness against model poisoning attacks in federated learning from a client perspective." Advances in Neural Information Processing Systems 34 (2021): 12613-12624.

defense @ client
overview: this is a client-based defense named White Blood Cell (WBC), which identifies the parameter space where long-lasting attack effect
          on parameters resides and perturb that space during local training.

Steps:
(1) Initialize two addition model parameters W_1 and W_2, and randomly generated a Laplace noise matrix M with mean = 0 and std = s (noise)
(2) Let W_1 be the original parameter W and update W = W - eta * gridient
(3) In the training, for each non-first batch, update W* = (W - W_1) - (W_1 - W_2)
(4) For each element that |W*| - eta * M <= 0, update the correspoinding elements via W = W + eta * M
"""


class WbcDefense(BaseDefenseMethod):
    """
    Weight-Based Client Defense for Federated Learning.

    This defense method performs weight-based client defense for federated learning.

    Args:
        args: Argument object containing client and batch indices.

    Methods:
        run(
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            base_aggregation_func: Callable = None,
            extra_auxiliary_info: Any = None,
        ) -> Dict:
        Run the weight-based client defense.

    Attributes:
        args: Argument object containing client and batch indices.
        client_idx: Index of the client.
        batch_idx: Index of the batch.
        old_gradient: Dictionary to store old gradients for weight perturbation.
    """

    def __init__(self, args):
        """
        Initialize the WbcDefense.

        Args:
            args: Argument object containing client and batch indices.
        """
        self.args = args
        self.client_idx = args.client_idx
        self.batch_idx = args.batch_idx
        self.old_gradient = {}

    def run(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ) -> Dict:
        """
        Run the weight-based client defense.

        Args:
            raw_client_grad_list (List[Tuple[float, OrderedDict]]):
                List of tuples containing client gradients as OrderedDict.
            base_aggregation_func (Callable, optional):
                Base aggregation function (currently unused).
            extra_auxiliary_info (Any, optional):
                Extra auxiliary information (currently unused).

        Returns:
            Dict: Dictionary containing aggregated model parameters.
        """
        num_client = len(raw_client_grad_list)
        vec_local_w = [
            (
                raw_client_grad_list[i][0],
                utils.vectorize_weight(raw_client_grad_list[i][1]),
            )
            for i in range(0, num_client)
        ]

        # extra auxiliary information: model parameters at current round -> dict
        models_param = extra_auxiliary_info
        model_param = models_param[self.client_idx][1]

        new_model_param = {}
        if self.batch_idx != 0:
            for (k, v) in model_param.items():
                if "weight" in k:
                    grad_tensor = (
                        raw_client_grad_list[self.client_idx][1][k].cpu(
                        ).numpy()
                    )
                    # for testing, simply pre-defin old gradient
                    self.old_gradient[k] = grad_tensor * 0.2
                    grad_diff = grad_tensor - self.old_gradient[k]
                    pert_strength = 1
                    pertubation = np.random.laplace(
                        0, pert_strength, size=grad_tensor.shape
                    ).astype(np.float32)
                    pertubation = np.where(
                        abs(grad_diff) > abs(pertubation), 0, pertubation
                    )
                    learning_rate = 0.1
                    new_model_param[k] = torch.from_numpy(
                        model_param[k].cpu().numpy() +
                        pertubation * learning_rate
                    )
                else:
                    new_model_param[k] = model_param[k]
        for (k, v) in model_param.items():
            if "weight" in k:
                self.old_gradient[k] = (
                    raw_client_grad_list[self.client_idx][1][k].cpu().numpy()
                )

        param_list = []
        for i in range(0, num_client):
            if i != self.client_idx or self.batch_idx == 0:
                param_list.append(models_param[i])
            else:
                param_list.append(
                    (models_param[self.client_idx][0], new_model_param))
                logging.info(f"New. param: {param_list[i]}")

        return base_aggregation_func(self.args, param_list)  # avg_params
