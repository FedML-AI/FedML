from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision import transforms
from fedml.model.cv.resnet import resnet56
from .attack_base import BaseAttackMethod
from ..common.net import LeNet
from ..common.utils import cross_entropy_for_onehot
from typing import Any

"""
ref: Zhu, Ligeng, Zhijian Liu, and Song Han. "Deep leakage from gradients." Advances in neural information processing systems 32 (2019).
attack @ (malicious) server

Steps:
(1) At the very beginning, the malicious server sends initialized parameters to clients and receives the local gradients from one client.
(2) The adversary recovery original data of the client via minimizing arg min_x* || g(x,y) - g(x*, y) ||, which is solve by an L-BFGS solver.
"""


class DLGAttack(BaseAttackMethod):
    def __init__(self, args):
        if args.model == "LeNet":
            self.model = LeNet()
        elif args.model == "resnet56":
            self.model = resnet56(10, pretrained=True, path=None)
        else:
            raise Exception(f"do not support this model: {args.model}")
        if args.dataset in ["cifar10", "cifar100"]:
            self.original_data_size = torch.Size([1, 3, 32, 32])
            if args.dataset == "cifar10":
                self.original_label_size = torch.Size([1, 10])
            else:
                self.original_label_size = torch.Size([1, 100])
        elif args.dataset == "mnist":
            self.original_data_size = torch.Size([1, 28, 28])
            self.original_label_size = torch.Size([1, 10])
        else:
            raise Exception(f"do not support this dataset for DLG attack: {args.dataset}")
        self.criterion = cross_entropy_for_onehot
        # cifar 100：
        # original data size = torch.Size([1, 3, 32, 32])
        # original label size = torch.Size([1, 100])
        self.attack_round_num = args.attack_round_num

        # extension: we allow users to protect some layers, which means the adversary can not get access to such layers
        if hasattr(args, "protected_layers"):
            self.protected_layers = args.protected_layers  # it should be a list of integers
        else:
            self.protected_layers = None  # no attack

    def reconstruct_data(self, a_model: OrderedDict, extra_auxiliary_info: Any = None):
        global_model_of_last_round = extra_auxiliary_info
        # gradient = OrderedDict()
        # (weight_name, importance_feature) = list(a_model.items())[-2]
        gradient = []
        protected_keys = []
        all_keys = list(global_model_of_last_round.keys())
        if self.protected_layers is not None:
            for protected_layer_idx in self.protected_layers:
                protected_keys.append(all_keys[protected_layer_idx])
        for key in global_model_of_last_round.keys():
            if key in protected_keys:
                gradient.append(torch.from_numpy(np.zeros(global_model_of_last_round[key].size())).float())
                # if the layer is protected, set to 0
            else:
                gradient.append(a_model[key] - global_model_of_last_round[key])  # todo: to double check
        gradient = tuple(gradient)
        dummy_data = torch.randn(self.original_data_size)
        dummy_label = torch.randn(self.original_label_size)
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
        tt = transforms.ToPILImage()

        history = []
        for iters in range(self.attack_round_num):
            def closure():
                optimizer.zero_grad()
                dummy_pred = self.model(dummy_data)
                dummy_loss = self.criterion(dummy_pred, F.softmax(dummy_label, dim=-1))
                dummy_grad = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)
                grad_diff = 0
                for gx, gy in zip(dummy_grad, gradient):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                return grad_diff

            optimizer.step(closure)

            # print(f"dummy_data = {dummy_data}")
            # print(f"dummy_label = {dummy_label}")
            if iters % 10 == 0:
                current_loss = closure()
                print(iters, "%.4f" % current_loss.item())
                history.append(tt(dummy_data[0].cpu()))

        plt.figure(figsize=(12, 8))
        for i in range(len(history)):
            plt.subplot(3, 10, i + 1)
            plt.imshow(history[i])
            plt.title("iter=%d" % (i * 10))
            plt.axis('off')

        plt.show()

        return dummy_data, dummy_label


        # self.data_size, self.attack_label, self.num_class = extra_auxiliary_info
        # # generate dummy data and label
        # dummy_data = torch.randn(self.data_size).requires_grad_(True)
        # dummy_label = torch.randn([1, self.num_class]).requires_grad_(True)
        # logging.info("Ground truth label is %s." % self.attack_label)
        # logging.info("Dummy label is %s." % torch.argmax(dummy_label, dim=-1).item())
        #
        # optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
        # original_dy_dx = extra_auxiliary_info
        #
        # # begin attack!
        # for iters in range(self.attack_round_num):
        #
        #     def closure():
        #         optimizer.zero_grad()
        #
        #         pred = self.model(dummy_data)
        #         dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        #         dummy_loss = cross_entropy_for_onehot(
        #             pred, dummy_onehot_label
        #         )  # TODO: fix the gt_label to dummy_label in both code and slides.
        #         dummy_dy_dx = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)
        #
        #         grad_diff = 0
        #         grad_count = 0
        #         for gx, gy in zip(dummy_dy_dx, original_dy_dx):  # TODO: fix the variablas here
        #             print(f"gx = {gx}")
        #             print(f"gy = {gy}")
        #             grad_diff += ((gx - gy) ** 2).sum()
        #             grad_count += gx.nelement()
        #         # grad_diff = grad_diff / grad_count * 1000
        #         grad_diff.backward()
        #
        #         return grad_diff
        #
        #     optimizer.step(closure)
        #     if iters % 10 == 0:
        #         current_loss = closure()
        #         print(iters, "%.4f" % current_loss.item())
        #
        # logging.info("Ground truth label is %s." % self.attack_label)
        # logging.info("After DLG, Dummy label is %s." % torch.argmax(dummy_label, dim=-1).item())
        # if self.attack_label == torch.argmax(dummy_label, dim=-1).item():
        #     # logging.info("The DLG attack client %s succeeds!" % self.attack_client_idx)
        #     logging.info("The DLG attack succeeds!")
        # else:
        #     # logging.info("The DLG attack client %s fails!" % self.attack_client_idx)
        #     logging.info("The DLG attack fails!")
