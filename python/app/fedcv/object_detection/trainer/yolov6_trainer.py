import copy
import logging
import time
import sys
import os
from pathlib import Path
import math
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim import Adam, lr_scheduler

import fedml
from fedml.core.alg_frame.client_trainer import ClientTrainer

from model.yolov6.yolov6.models.loss import ComputeLoss
from utils.general import (
    box_iou,
    xywh2xyxy,
    clip_coords,
)
from model.yolov6.yolov6.utils.nms import non_max_suppression
from utils.metrics import ap_per_class


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

    def train(self, train_data, device, args=None):
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

    def test(self, test_data, device, args):
        logging.info("Evaluating on Trainer ID: {}".format(self.id))
        model = self.model
        self.round_idx = args.round_idx
        args = self.args

        test_metrics = {
            "test_correct": 0,
            "test_total": 0,
            "test_loss": 0,
        }

        if not test_data:
            logging.info("No test data for this trainer")
            return test_metrics

        # model.train()
        # torch.set_grad_enabled(False)
        model.eval()
        model.to(device)

        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()

        seen = 0
        names = {
            k: v
            for k, v in enumerate(
                model.names if hasattr(model, "names") else model.module.names
            )
        }
        s = ("%20s" + "%12s" * 6) % (
            "Class",
            "Images",
            "Targets",
            "P",
            "R",
            "mAP@.5",
            "mAP@.5:.95",
        )
        p, r, f1, mp, mr, map50, map, t0, t1 = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        # loss = torch.zeros(4, device=device)
        # compute_loss = ComputeLoss(iou_type=args.yolov6_cfg.model.head.iou_type)
        jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

        for batch_i, (img, targets, paths, shapes) in enumerate(test_data):
            img = img.to(device, non_blocking=True)
            # img = img.half() if half else img.float()  # uint8 to fp16/32
            img = img.float() / 256.0 - 0.5
            targets = targets.to(device)
            nb, _, height, width = img.shape  # batch size, channels, height, width
            whwh = torch.Tensor([width, height, width, height]).to(device)

            # Disable gradients
            with torch.no_grad():
                # Run model
                outputs = model(img)  # inference and training outputs
                # print(f"image shape: {img.shape}, output shape: {outputs[0].shape}")

                # Run NMS
                output = non_max_suppression(
                    outputs, conf_thres=args.conf_thres, iou_thres=args.iou_thres
                )

            # Statistics per image
            for si, pred in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append(
                            (
                                torch.zeros(0, niou, dtype=torch.bool),
                                torch.Tensor(),
                                torch.Tensor(),
                                tcls,
                            )
                        )
                    continue

                # W&B logging
                # TODO

                # Clip boxes to image bounds
                clip_coords(pred, (height, width))

                # Assign all predictions as incorrect
                correct = torch.zeros(
                    pred.shape[0], niou, dtype=torch.bool, device=device
                )
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (
                            (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                        )  # prediction indices
                        pi = (
                            (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)
                        )  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(pred[pi, :4], tbox[ti]).max(
                                1
                            )  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if (
                                        len(detected) == nl
                                    ):  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

            # Plot images
            # TODO

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(
                *stats, plot=False, save_dir=args.save_dir, names=names
            )
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(
                stats[3].astype(np.int64), minlength=args.nc
            )  # number of targets per class
        else:
            nt = torch.zeros(1)

        # W&B logging
        # TODO
        # Print results
        pf = "%20s" + "%12.3g" * 6  # print format
        print(pf % ("all", seen, nt.sum(), mp, mr, map50, map))

        # Print results per class
        if args.yolo_verbose and args.nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Print speeds
        t = tuple(x / seen * 1e3 for x in (t0, t1, t0 + t1)) + (
            args.img_size,
            args.img_size,
            args.batch_size,
        )  # tuple

        maps = np.zeros(args.nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]

        # Return results

        model.train()
        model.float()  # for training

        # all metrics
        # metrics = (mp, mr, map50, map, *(loss.cpu() / len(test_data)).tolist()), maps, t
        # logging.info(f"Test metrics: {metrics}")

        fedml.mlops.log(
            {
                f"round_idx": self.round_idx,
                f"test_mp": np.float(mp),
                f"test_mr": np.float(mr),
                f"test_map50": np.float(map50),
                f"test_map": np.float(map),
                # f"test_loss": np.float(sum((loss.cpu() / len(test_data)).tolist())),
            }
        )

        test_metrics = {
            "test_correct": ap_class,
            "test_total": len(test_data),
            "test_loss": map,
        }
        logging.info(f"Test metrics: {test_metrics}")
        return test_metrics

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ):
        return False
        logging.info("Testing on the server")
        logging.info(f"rank id: {args.rank}")
        train_data = train_data_local_dict
        test_data = test_data_local_dict

        for k in train_data.keys():
            # logging.info(f"{k}: {train_data[k]}")
            if train_data[k] is None:
                continue
            train_data_results = self.test(
                test_data=train_data[k], device=device, args=args
            )
            logging.info(f"{k}: {train_data_results}")

        for k in test_data.keys():
            # logging.info(f"{k}: {test_data[k]}")
            if test_data[k] is None:
                continue
            test_data_results = self.test(
                test_data=test_data[k], device=device, args=args
            )
            logging.info(f"{k}: {test_data_results}")
        return True
