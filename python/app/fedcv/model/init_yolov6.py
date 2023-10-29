import logging
import os
import sys
from pathlib import Path
from warnings import warn

import torch
import yaml
import numpy as np
import glob
import os.path as osp

from data.data_loader_yolov6 import load_partition_data_coco
from trainer.yolov6_trainer import YOLOv6Trainer
from YOLOv6.yolov6.core.engine import Trainer
from YOLOv6.tools.train import get_args_parser, check_and_init
from model.util import EnsembleModel

try:
    import wandb
except ImportError:
    wandb = None
    logging.info(
        "Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)"
    )

def init_yolov6(args, device="cpu"):
    yolo_args = get_args_parser().parse_args()
    yolo_args.data_path = args.data_cfg
    yolo_args.conf_file = args.yolo_cfg
    yolo_args.img_size = args.img_size
    yolo_args.conf_file = args.yolo_cfg

    cfg, device, yolo_args = check_and_init(yolo_args)
    meituan_trainer = Trainer(yolo_args, cfg, device)
    # logging.info('Model: {}'.format(meituan_trainer.model))
    ensemble_model = EnsembleModel(meituan_trainer.model, meituan_trainer.ema.ema)

    dataset, net_dataidx_map = load_partition_data_coco(args, yolo_args, cfg, device)

    trainer = YOLOv6Trainer(ensemble_model, args, yolo_args, cfg, net_dataidx_map)

    return ensemble_model, dataset, trainer, args, yolo_args, cfg
