import copy
import logging
import math
import time

import yaml

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import fedml
from fedml.core import ClientTrainer
from model.yolov5.utils.loss import ComputeLoss
from model.yolov5.utils.general import non_max_suppression, xywh2xyxy, scale_coords
from model.yolov5.utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from model.yolov5.val import run as run_val


class YOLOv5Trainer(ClientTrainer):
    def __init__(self, model, args=None):
        super(YOLOv5Trainer, self).__init__(model, args)
        self.hyp = args.hyp
        self.args = args
        self.round_loss = []
        self.round_idx = 0

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

        compute_loss = ComputeLoss(model)

        epoch_loss = []
        mloss = torch.zeros(3, device=device)  # mean losses
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
                # with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(
                    pred, targets.to(device).float()
                )  # loss scaled by batch_size

                # Backward
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

                mloss = (mloss * batch_idx + loss_items) / (
                    batch_idx + 1
                )  # update mean losses
                mem = "%.3gG" % (
                    torch.cuda.memory_reserved() / 1e9
                    if torch.cuda.is_available()
                    else 0
                )  # (GB)
                s = ("%10s" * 2 + "%10.4g" * 5) % (
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
                f"Trainer {self.id} epoch {epoch} box: {mloss[0]} obj: {mloss[1]} cls: {mloss[2]} total: {mloss.sum()} time: {(time.time() - t)}"
            )

            logging.info("#" * 20)

            logging.info(
                f"Trainer {self.id} epoch {epoch} time: {(time.time() - t)}s batch_num: {batch_idx} speed: {(time.time() - t)/batch_idx} s/batch"
            )
            logging.info("#" * 20)

            if (epoch + 1) % self.args.checkpoint_interval == 0:
                model_path = (
                    self.args.save_dir
                    / "weights"
                    / f"model_client_{self.id}_epoch_{epoch}.pt"
                )
                logging.info(
                    f"Trainer {self.id} epoch {epoch} saving model to {model_path}"
                )
                torch.save(model.state_dict(), model_path)

            if (epoch + 1) % self.args.frequency_of_the_test == 0:
                logging.info("Start val on Trainer {}".format(self.id))
                self.val(model, train_data, device, args)

        logging.info("End training on Trainer {}".format(self.id))
        torch.save(
            model.state_dict(),
            self.args.save_dir / "weights" / f"model_client_{self.id}.pt",
        )

        # plot for client
        # plot box, obj, cls, total loss
        epoch_loss = np.array(epoch_loss)
        # logging.info(f"Epoch loss: {epoch_loss}")

        fedml.mlops.log(
            {
                f"round_idx": self.round_idx,
                f"train_box_loss": np.float(epoch_loss[-1, 0]),
                f"train_obj_loss": np.float(epoch_loss[-1, 1]),
                f"train_cls_loss": np.float(epoch_loss[-1, 2]),
                f"train_total_loss": np.float(epoch_loss[-1, :].sum()),
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

    def val(self, model, train_data, device, args):
        logging.info(f"Trainer {self.id} val start")
        self.round_idx = args.round_idx
        args = self.args
        hyp = self.hyp if self.hyp else self.args.hyp

        model.eval()
        model.to(device)
        compute_loss = ComputeLoss(model)

        # val
        t = time.time()
        with open(args.data_conf) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict

        results, maps, _ = run_val(
            data_dict,
            batch_size=args.batch_size,
            imgsz=args.img_size,
            model=model,
            single_cls=args.single_cls,
            dataloader=train_data,
            save_dir=args.save_dir,
            plots=False,
            compute_loss=compute_loss,
        )
        logging.info(results)
        mp, mr, map50, map, box_loss, obj_loss, cls_loss = results
        logging.info(
            f"Trainer {self.id} val time: {(time.time() - t)}s speed: {(time.time() - t)/len(train_data)} s/batch"
        )
        logging.info(
            f"Trainer {self.id} val box: {box_loss} obj: {obj_loss} cls: {cls_loss} total: {box_loss + obj_loss + cls_loss}"
        )
        logging.info(
            f"Trainer {self.id} val map: {map} map50: {map50} mp: {mp} mr: {mr}"
        )

        return
