import logging
import torch
import torch.nn.functional as F
from .attack_base import BaseAttackMethod
from ..common.utils import cross_entropy_for_onehot
from typing import Any

logging.getLogger().setLevel(logging.INFO)

"""
ref: Zhu, Ligeng, Zhijian Liu, and Song Han. "Deep leakage from gradients." Advances in neural information processing systems 32 (2019).
attack @ (malicious) server

Steps:
(1) At the very beginning, the malicious server sends initialized parameters to clients and receives the local gradients from one client.
(2) The adversary recovery original data of the client via minimizing arg min_x* || g(x,y) - g(x*, y) ||, which is solve by an L-BFGS solver.
"""


class DLGAttack(BaseAttackMethod):
    def __init__(self, model, attack_epoch):
        # self.attack_client_idx = attack_client_idx # todo: how to determine which clients to attack??
        self.model = model
        self.attack_epoch = attack_epoch  # todo: discuss with chaoyang

    def reconstruct_data(self, a_gradient, extra_auxiliary_info: Any = None):
        self.data_size, self.attack_label, self.num_class = extra_auxiliary_info
        # generate dummy data and label
        dummy_data = torch.randn(self.data_size).requires_grad_(True)
        dummy_label = torch.randn([1, self.num_class]).requires_grad_(True)
        logging.info("Ground truth label is %s." % self.attack_label)
        logging.info("Dummy label is %s." % torch.argmax(dummy_label, dim=-1).item())

        optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
        original_dy_dx = extra_auxiliary_info

        # begin attack!
        for iters in range(self.attack_epoch):

            def closure():
                optimizer.zero_grad()

                pred = self.model(dummy_data)
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = cross_entropy_for_onehot(
                    pred, dummy_onehot_label
                )  # TODO: fix the gt_label to dummy_label in both code and slides.
                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)

                grad_diff = 0
                grad_count = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx):  # TODO: fix the variablas here
                    print(f"gx = {gx}")
                    print(f"gy = {gy}")
                    grad_diff += ((gx - gy) ** 2).sum()
                    grad_count += gx.nelement()
                # grad_diff = grad_diff / grad_count * 1000
                grad_diff.backward()

                return grad_diff

            optimizer.step(closure)
            if iters % 10 == 0:
                current_loss = closure()
                print(iters, "%.4f" % current_loss.item())

        logging.info("Ground truth label is %s." % self.attack_label)
        logging.info("After DLG, Dummy label is %s." % torch.argmax(dummy_label, dim=-1).item())
        if self.attack_label == torch.argmax(dummy_label, dim=-1).item():
            # logging.info("The DLG attack client %s succeeds!" % self.attack_client_idx)
            logging.info("The DLG attack succeeds!")
        else:
            # logging.info("The DLG attack client %s fails!" % self.attack_client_idx)
            logging.info("The DLG attack fails!")
