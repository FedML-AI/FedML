from collections import OrderedDict
import torch
from .defense_base import BaseDefenseMethod
from ..common import utils
from typing import Callable, List, Tuple, Dict, Any

"""
defense, added by Shanshan, 06/28/2022
"Can You Really Backdoor Federated Learning?" 
https://arxiv.org/pdf/1911.07963.pdf 
"""


class NormDiffClippingDefense(BaseDefenseMethod):
    def __init__(self, config):
        """
        Initialize the NormDiffClippingDefense method.

        Args:
            config (object): Configuration object containing defense parameters.
        """
        self.config = config
        # for norm diff clipping; in the paper, they set it to 0.1, 0.17, and 0.33.
        self.norm_bound = config.norm_bound

    def defend_before_aggregation(
            self,
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            extra_auxiliary_info: Any = None,
    ):
        """
        Perform defense before aggregation using norm difference clipping.

        Args:
            raw_client_grad_list (List[Tuple[float, OrderedDict]]): List of client gradients.
            extra_auxiliary_info (Any): Global model for clipping (optional).

        Returns:
            List[Tuple[float, OrderedDict]]: List of defended client gradients.
        """
        global_model = extra_auxiliary_info
        vec_global_w = utils.vectorize_weight(global_model)
        new_grad_list = []
        for (sample_num, local_w) in raw_client_grad_list:
            vec_local_w = utils.vectorize_weight(local_w)
            clipped_weight_diff = self._get_clipped_norm_diff(
                vec_local_w, vec_global_w)
            clipped_w = self._get_clipped_weights(
                local_w, global_model, clipped_weight_diff
            )
            new_grad_list.append((sample_num, clipped_w))
        return new_grad_list

    def _get_clipped_norm_diff(self, vec_local_w, vec_global_w):
        """
        Compute the clipped norm difference between local and global weights.

        Args:
            vec_local_w (torch.Tensor): Vectorized local weights.
            vec_global_w (torch.Tensor): Vectorized global weights.

        Returns:
            torch.Tensor: Clipped weight difference.
        """
        vec_diff = vec_local_w - vec_global_w
        weight_diff_norm = torch.norm(vec_diff).item()
        clipped_weight_diff = vec_diff / \
            max(1, weight_diff_norm / self.norm_bound)
        return clipped_weight_diff

    @staticmethod
    def _get_clipped_weights(local_w, global_w, weight_diff):
        """
        Compute clipped weights based on global and local weights.

        Args:
            local_w (OrderedDict): Local model weights.
            global_w (OrderedDict): Global model weights.
            weight_diff (torch.Tensor): Clipped weight difference.

        Returns:
            OrderedDict: Clipped local model weights.
        """

        #  rule: global_w + clipped(local_w - global_w)
        recons_local_w = OrderedDict()
        index_bias = 0
        for item_index, (k, v) in enumerate(local_w.items()):
            if utils.is_weight_param(k):
                recons_local_w[k] = (
                    weight_diff[index_bias: index_bias +
                                v.numel()].view(v.size())
                    + global_w[k]
                )
                index_bias += v.numel()
            else:
                recons_local_w[k] = v
        return recons_local_w
