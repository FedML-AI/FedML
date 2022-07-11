import random
import numpy as np
import torch
import torch.nn.functional as F
from fedml.core.security.attack.attack_base import BaseAttackMethod

# test using local directory
# from security.attack.attack_base import BaseAttackMethod

"""
ref: Zhu, Ligeng, Zhijian Liu, and Song Han. "Deep leakage from gradients." Advances in neural information processing systems 32 (2019).
attack @ server, added by Kai, 07/06/2022

Steps:
(1) At the very beginning, the malicious server sends initialized parameters to clients and receives the local gradients from one client.
(2) The adversary recovery original data of the client via minimizing arg min_x* || g(x,y) - g(x*, y) ||, which is solve by an L-BFGS solver.
"""


class DLGAttack(BaseAttackMethod):
    def __init__(
        self, data_size, num_class, model, criterion, attack_epoch, attack_label=84
    ):
        self.data_size = [
            1
        ] + data_size  # batched image size (list), e.g. [1, 3, 32, 32]
        self.num_class = num_class  # number of classess of the dataset
        self.model = model
        self.criterion = criterion
        self.attack_epoch = attack_epoch
        self.attack_label = attack_label

    def attack(self, local_w, global_w, refs=None):
        # generate dummy data and label
        dummy_data = torch.randn(self.data_size).requires_grad_(True)
        dummy_label = torch.randn([1, self.num_class]).requires_grad_(True)
        print(
            "Ground truth label is %d." % self.attack_label,
            "Dummy label is %d." % torch.argmax(dummy_label, dim=-1).item(),
        )

        optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
        # load local model
        self.model.load_state_dict(local_w, strict=True)
        original_dy_dx = refs

        # begin attack!
        for iters in range(self.attack_epoch):

            def closure():
                optimizer.zero_grad()

                pred = self.model(dummy_data)
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = self.criterion(
                    pred, dummy_onehot_label
                )  # TODO: fix the gt_label to dummy_label in both code and slides.
                dummy_dy_dx = torch.autograd.grad(
                    dummy_loss, self.model.parameters(), create_graph=True
                )

                grad_diff = 0
                grad_count = 0
                for gx, gy in zip(
                    dummy_dy_dx, original_dy_dx
                ):  # TODO: fix the variablas here
                    grad_diff += ((gx - gy) ** 2).sum()
                    grad_count += gx.nelement()
                # grad_diff = grad_diff / grad_count * 1000
                grad_diff.backward()

                return grad_diff

            optimizer.step(closure)
            if iters % 10 == 0:
                current_loss = closure()
                print(iters, "%.4f" % current_loss.item())

        print(
            "Ground truth label is %d." % self.attack_label,
            "After DLG, Dummy label is %d." % torch.argmax(dummy_label, dim=-1).item(),
        )
        # if self.attack_label == torch.argmax(dummy_label, dim=-1).item():
        #     print("The DLG attack is succesfful!")
