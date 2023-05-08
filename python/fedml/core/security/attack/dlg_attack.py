from collections import OrderedDict
# import time
import numpy as np
import torch
import torch.nn.functional as F
# from matplotlib import pyplot as plt
from torchvision import transforms
from typing import List, Tuple, Any
import fedml
from fedml.model.cv.resnet import resnet56, resnet20
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
        self.model = None
        self.model_type = args.model


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
        self.device = fedml.device.get_device(args)

        # extension: we allow users to protect some layers, which means the adversary can not get access to such layers
        if hasattr(args, "protected_layers"):
            self.protected_layers = args.protected_layers  # it should be a list of integers
        else:
            self.protected_layers = None  # no protected layers
        self.iteration_num = 0
        if hasattr(args, "attack_iteration_idxs"):
            self.attack_iteration_idxs = args.attack_iteration_idxs
        else:
            self.attack_iteration_idxs = [args.comm_round - 1]
            # attack the last iteration, as it contains more information

    def get_model(self):
        if self.model_type == "LeNet":
            return LeNet()
        elif self.model_type == "resnet56":
            return resnet56(10, pretrained=False, path=None)
        elif self.model_type == "resnet20":
            return resnet20(10)
        else:
            raise Exception(f"do not support this model: {self.model_type}")

    def reconstruct_data(self, raw_client_grad_list: List[Tuple[float, OrderedDict]], extra_auxiliary_info: Any = None):
        if self.iteration_num in self.attack_iteration_idxs:
            for (_, local_model) in raw_client_grad_list:
                print(f"-----------attack---------------")
                self.reconstruct_data_using_a_model(a_model=local_model, extra_auxiliary_info=extra_auxiliary_info)
        self.iteration_num += 1

    def reconstruct_data_using_a_model(self, a_model: OrderedDict, extra_auxiliary_info: Any = None):
        self.model = self.get_model()
        global_model_of_last_round = extra_auxiliary_info
        gradient = []

        layer_counter = 0
        for k, _ in global_model_of_last_round.items():
            if "weight" in k or "bias" in k:
                if self.protected_layers is not None and layer_counter in self.protected_layers:
                    gradient.append(torch.from_numpy(np.zeros(global_model_of_last_round[k].size())).float())
                    # if the layer is protected, set to 0
                else:
                    gradient.append(a_model[k] - global_model_of_last_round[k].to(self.device))  # !!!!!!!!!!!!!!!!!!todo: to double check
                layer_counter += 1
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
                dummy_grad = tuple(
                    g.to(self.device) for g in dummy_grad)  # extract tensor from tuple and move to device

                grad_diff = 0
                # print(f"len(dummy_grad) = {len(dummy_grad)}, len(gradient) = {len(gradient)}")
                # print(f"gradient={gradient}")
                # print(f"dummy_grad={dummy_grad}")
                for gx, gy in zip(dummy_grad, gradient):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                return grad_diff

            optimizer.step(closure)

            # print(f"dummy_data = {dummy_data}")
            # print(f"dummy_label = {dummy_label}")
            if iters % 50 == 0:
                current_loss = closure()
                print(iters, "%.4f" % current_loss.item())
                history.append(tt(dummy_data[0].cpu()))

        # plot figures:
        # plt.figure(figsize=(12, 8))
        # for i in range(len(history)):
        #     plt.subplot(3, 10, i + 1)
        #     plt.imshow(history[i])
        #     plt.title("iter=%d" % (i * 50))
        #     plt.axis('off')
        # # plt.show()
        # filename = str(self.iteration_num) + '_' + str(round(time.time()*1000)) + '.png'
        # plt.savefig(filename)

        return dummy_data, dummy_label
