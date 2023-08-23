import logging
import math
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from .attack_base import BaseAttackMethod
import numpy as np
from ..common.utils import (
    replace_original_class_with_target_class,
    get_client_data_stat,
)

"""
ref: Tolpegin, Vale, Truex,  "Data Poisoning Attacks Against Federated Learning Systems."  (2021).
attack @client, added by Yuhui, 07/08/2022
"""


class LabelFlippingAttack(BaseAttackMethod):
    def __init__(self, args):
        self.original_class_list = args.original_class_list
        self.target_class_list = args.target_class_list
        self.batch_size = args.batch_size
        self.poisoned_client_list = []
        self.ite_counter = 0
        if hasattr(args, "poison_start_round_id") and isinstance(args.poison_start_round_id, int):
            self.poison_start_round_id = args.poison_start_round_id
        else:
            self.poison_start_round_id = 0
        if hasattr(args, "poison_end_round_id") and isinstance(args.poison_end_round_id, int):
            self.poison_end_round_id = args.poison_end_round_id
        else:
            self.poison_end_round_id = args.comm_round - 1
        if hasattr(args, "ratio_of_poisoned_client") and isinstance(args.ratio_of_poisoned_client, float):
            if args.ratio_of_poisoned_client < 0 or args.ratio_of_poisoned_client > 1:
                raise Exception("unknown ratio_of_poisoned_client")
            self.ratio_of_poisoned_client = args.ratio_of_poisoned_client
        else:
            raise Exception("unknown poisoned client number")

        self.client_num_per_round = args.client_num_per_round
        self.counter = 0

    def get_ite_num(self):
        return math.floor(self.counter / self.client_num_per_round)  # ite num starts from 0

    def is_to_poison_data(self):
        self.counter += 1
        if self.get_ite_num() < self.poison_start_round_id or self.get_ite_num() > self.poison_end_round_id:
            return False
        np.random.seed(self.counter)
        rand = np.random.random()
        # rand = random.random()
        return rand < self.ratio_of_poisoned_client

    def print_dataset(self, dataset):
        print("---------------print dataset------------")
        for batch_idx, (data, target) in enumerate(dataset):
            print(f"{batch_idx} ----- {target}")

    def poison_data(self, local_dataset):
        get_client_data_stat(local_dataset)
        # print("=======================1 end ")
        # self.print_dataset(local_dataset)
        # get_client_data_stat(local_dataset)
        # print("======================= 2 end")
        tmp_local_dataset_x = torch.Tensor([])
        tmp_local_dataset_y = torch.Tensor([])
        targets_set = {}
        for batch_idx, (data, targets) in enumerate(local_dataset):
            tmp_local_dataset_x = torch.cat((tmp_local_dataset_x, data))
            tmp_local_dataset_y = torch.cat((tmp_local_dataset_y, targets))

            for t in targets.tolist():
                if t in targets_set.keys():
                    targets_set[t] += 1
                else:
                    targets_set[t] = 1
        total_counter = 0
        for item in targets_set.items():
            # print("------target:{} num:{}".format(item[0], item[1]))
            total_counter += item[1]
        # print(f"total counter = {total_counter}")

        ####################### below are correct ###############################3

        tmp_y = replace_original_class_with_target_class(
            data_labels=tmp_local_dataset_y,
            original_class_list=self.original_class_list,
            target_class_list=self.target_class_list,
        )
        dataset = TensorDataset(tmp_local_dataset_x, tmp_y)
        poisoned_data = DataLoader(dataset, batch_size=self.batch_size)
        get_client_data_stat(poisoned_data)

        return poisoned_data