import copy
import logging
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import fedml
from fedml.core.alg_frame.client_trainer import ClientTrainer
from model.yolov6.yolov6.models.loss import ComputeLoss


class YOLOv6Trainer(ClientTrainer):
    def __init__(self, model, args=None):
        super(YOLOv6Trainer, self).__init__(model, args)
        self.hyp = args.hyp
        self.args = args
        self.round_idx = 0
        self.round_loss = []

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        logging.info("Start training on Trainer {}".format(self.id))
        logging.info(f"Hyperparameters: {self.hyp}, Args: {self.args}")
        model = self.model
        self.round_idx = args.round_idx
        args = self.args
        hyp = self.hyp if self.hyp else self.args.hyp

        epochs = args.epochs  # number of epochs

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
        if args.client_optimizer == "adam":
            optimizer = optim.Adam(
                pg0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999)
            )  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(
                pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True
            )

        optimizer.add_param_group(
            {"params": pg1, "weight_decay": hyp["weight_decay"]}
        )  # add pg1 with weight_decay
        optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
        logging.info(
            "Optimizer groups: %g .bias, %g conv.weight, %g other"
            % (len(pg2), len(pg1), len(pg0))
        )
        del pg0, pg1, pg2

        # Freeze
        freeze = []  # parameter names to freeze (full or partial)
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print("freezing %s" % k)
                v.requires_grad = False

        total_epochs = epochs * args.comm_round

        lf = (
            lambda x: ((1 + math.cos(x * math.pi / total_epochs)) / 2)
            * (1 - hyp["lrf"])
            + hyp["lrf"]
        )  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        model.to(device)
        model.train()

        scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1.0 - epoch / args.epochs
        )
        compute_loss = ComputeLoss(iou_type=self.args.yolov6_cfg.model.head.iou_type)

        epoch_loss = []
        mloss = torch.zeros(4, device=device)  # mean losses
        logging.info("Epoch gpu_mem box obj cls total targets img_size time")
        for epoch in range(args.epochs):
            model.train()
            t = time.time()
            batch_loss = []
            logging.info("Trainer_ID: {0}, Epoch: {1}".format(self.id, epoch))

            for (batch_idx, batch) in enumerate(train_data):
                imgs, targets, paths, _ = batch
                imgs = imgs.to(device, non_blocking=True).float() / 256.0 - 0.5

                optimizer.zero_grad()
                pred = model(imgs)  # forward
                # print(f"image shape: {imgs.shape}, pred shape: {pred[0].shape}")
                # print("shape:", pred[0].shape)
                total_loss, loss_items = compute_loss(
                    pred, targets.to(device)
                )  # loss scaled by batch_size

                # Backward
                total_loss.backward()
                optimizer.step()
                batch_loss.append(total_loss.item())

                mloss = (mloss * batch_idx + loss_items) / (
                    batch_idx + 1
                )  # update mean losses
                mem = "%.3gG" % (
                    torch.cuda.memory_reserved() / 1e9
                    if torch.cuda.is_available()
                    else 0
                )  # (GB)
                s = ("%10s" * 3 + "%10.4g" * 6) % (
                    self.id,
                    "%g/%g" % (epoch, epochs - 1),
                    mem,
                    *mloss,
                    targets.shape[0],
                    imgs.shape[-1],
                )
                logging.info(s)

            scheduler.step()

            epoch_loss.append(copy.deepcopy(mloss.cpu().numpy()))
            logging.info(
                f"Trainer {self.id} epoch {epoch} box: {mloss[0]} obj: {mloss[1]} cls: {mloss[2]} total: {mloss[3]} time: {(time.time() - t)/60}"
            )

            logging.info("#" * 20)

            logging.info(
                f"Trainer {self.id} epoch {epoch} time: {(time.time() - t)}s batch_num: {batch_idx} speed: {(time.time() - t)/batch_idx} s/batch"
            )
            logging.info("#" * 20)

        # plot for client
        # plot box, obj, cls, total loss
        epoch_loss = np.array(epoch_loss)

        fedml.mlops.log(
            {
                f"round_idx": self.round_idx,
                f"train_box_loss": np.float(epoch_loss[-1, 0]),
                f"train_obj_loss": np.float(epoch_loss[-1, 1]),
                f"train_cls_loss": np.float(epoch_loss[-1, 2]),
                f"train_total_loss": np.float(epoch_loss[-1, 3]),
            }
        )

        self.round_loss.append(epoch_loss[-1, :])
        if self.round_idx == args.comm_round:
            self.round_loss = np.array(self.round_loss)
            # logging.info(f"round_loss shape: {self.round_loss.shape}")
            logging.info(
                f"Trainer {self.id} round {self.round_idx} finished, round loss: {self.round_loss}"
            )

        return
