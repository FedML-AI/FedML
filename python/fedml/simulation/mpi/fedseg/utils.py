import glob
import logging
import math
import os
import pickle
import shutil
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


# import coco_dataloader as dataloader
# import dataloader


def transform_list_to_tensor(model_params_list):
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(
            np.asarray(model_params_list[k])
        ).float()
    return model_params_list


def transform_tensor_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params


def save_as_pickle_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)
        f.close()


def load_from_pickle_file(path):
    return pickle.load(open(path, "rb"))


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def str_to_bool(s):
    if s == "True":
        return True
    elif s == "False":
        return False
    else:
        raise ValueError


class EvaluationMetricsKeeper:
    def __init__(self, accuracy, accuracy_class, mIoU, FWIoU, loss):
        self.acc = accuracy
        self.acc_class = accuracy_class
        self.mIoU = mIoU
        self.FWIoU = FWIoU
        self.loss = loss


# Segmentation Loss
class SegmentationLosses(object):
    def __init__(self, size_average=True, batch_average=True, ignore_index=255):
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.batch_average = batch_average

    def build_loss(self, mode="ce"):
        """Choices: ['ce' or 'focal']"""
        if mode == "ce":
            return self.CrossEntropyLoss
        elif mode == "focal":
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index, size_average=self.size_average
        )
        # if self.cuda:
        #     criterion = criterion.cuda()
        loss = criterion(logit, target.long())  # pylint: disable=E1102
        if self.batch_average:
            loss /= n
        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index, size_average=self.size_average
        )
        # if self.cuda:
        #     criterion = criterion.cuda()
        logpt = -criterion(logit, target.long())  # pylint: disable=E1102
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt
        if self.batch_average:
            loss /= n
        return loss


# LR Scheduler
class LR_Scheduler(object):
    """Learning Rate Scheduler
    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``
    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``
    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`,`step`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`
        iters_per_epoch: number of iterations per epoch
    """

    def __init__(
        self, mode, base_lr, num_epochs, iters_per_epoch=0, lr_step=0, warmup_epochs=0
    ):
        self.mode = mode
        logging.info("Using {} LR Scheduler!".format(self.mode))
        self.lr = base_lr
        if mode == "step":
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch):
        T = epoch * self.iters_per_epoch + i
        if self.mode == "cos":
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == "poly":
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == "step":
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplementedError
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]["lr"] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]["lr"] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]["lr"] = lr * 10


# save model checkpoints (centralized)
class Saver(object):
    def __init__(self, args):
        self.args = args
        self.directory = os.path.join("run", args.dataset, args.model, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, "experiment_*")))
        run_id = int(self.runs[-1].split("_")[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(
            self.directory, "experiment_{}".format(str(run_id))
        )
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename="checkpoint.pth.tar"):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state["best_pred"]
            with open(os.path.join(self.experiment_dir, "best_pred.txt"), "w") as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split("_")[-1]
                    path = os.path.join(
                        self.directory,
                        "experiment_{}".format(str(run_id)),
                        "best_pred.txt",
                    )
                    if os.path.exists(path):
                        with open(path, "r") as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(
                        filename, os.path.join(self.directory, "model_best.pth.tar")
                    )
            else:
                shutil.copyfile(
                    filename, os.path.join(self.directory, "model_best.pth.tar")
                )

    def save_experiment_config(self):

        logfile = os.path.join(self.experiment_dir, "parameters.txt")
        log_file = open(logfile, "w")

        p = OrderedDict()

        p["model"] = self.args.model
        p["backbone"] = self.args.backbone
        p["backbone_pretrained"] = self.args.backbone_pretrained
        p["backbone_freezed"] = self.args.backbone_freezed
        p["extract_feat"] = self.args.extract_feat
        p["outstride"] = self.args.outstride
        p["dataset"] = self.args.dataset
        p["partition_method"] = self.args.partition_method
        p["partition_alpha"] = self.args.partition_alpha
        p["client_num_in_total"] = self.args.client_num_in_total
        p["client_num_per_round"] = self.args.client_num_per_round
        p["batch_size"] = self.args.batch_size
        p["sync_bn"] = self.args.sync_bn
        p["freeze_bn"] = self.args.freeze_bn
        p["client_optimizer"] = self.args.client_optimizer
        p["lr"] = self.args.lr
        p["lr_scheduler"] = self.args.lr_scheduler
        p["momentum"] = self.args.momentum
        p["weight_decay"] = self.args.weight_decay
        p["nesterov"] = self.args.nesterov
        p["loss_type"] = self.args.loss_type
        p["epochs"] = self.args.epochs
        p["comm_round"] = self.args.comm_round
        p["evaluation_frequency"] = self.args.evaluation_frequency
        p["gpu_server_num"] = self.args.gpu_server_num
        p["gpu_num_per_server"] = self.args.gpu_num_per_server

        for key, val in p.items():
            log_file.write(key + ":" + str(val) + "\n")
        log_file.close()


# Evaluation Metrics
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype("int") + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
