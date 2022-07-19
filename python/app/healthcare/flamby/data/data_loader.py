import argparse
import copy
import os
import time

import numpy as np
import torch
from torch.optim import lr_scheduler


import flamby

from flamby.datasets.fed_kits19 import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedKits19,
    evaluate_dice_on_tests,
    metric,
    softmax_helper,
)
from flamby.utils import check_dataset_from_config
from .fed_kits19 import load_partition_fed_kits19


def load_data(args):
    dataset_name = args.dataset.lower()
    if dataset_name in ["fed_kits19", "fed_kits2019", "fed-kits2019", "fed-kits19"]:
        dataset = load_partition_fed_kits19(args)
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")

    return dataset
