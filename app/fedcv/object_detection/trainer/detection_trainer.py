from ast import Pass
import logging, time
import sys, os

import numpy as np
import torch
import wandb
from torch.optim import Adam, lr_scheduler

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from fedml.core.alg_frame.client_trainer import ClientTrainer
from ..model.yolo.utils.loss import ComputeLoss


class DetectionTrainer(ClientTrainer):
    def __init__(self, model, args=None):
        super(DetectionTrainer, self).__init__(model, args)

    def get_model_params(self):
        if self.args.backbone_freezed:
            logging.info("Initializing model; Backbone Freezed")
            return self.model.encoder_decoder.cpu().state_dict()
        else:
            logging.info("Initializing end-to-end model")
            return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        if self.args.backbone_freezed:
            logging.info("Updating Global model; Backbone Freezed")
            self.model.encoder_decoder.load_state_dict(model_parameters)
        else:
            logging.info("Updating Global model")
            self.model.load_state_dict(model_parameters)

    def train(self, train_data, device):
        model = self.model
        args = self.args
        model.to(device)
        model.train()
        criterion = ComputeLoss(model)

        if args.client_optimizer == "sgd":

            if args.backbone_freezed:
                optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=args.lr * 10,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    nesterov=args.nesterov,
                )
            else:
                train_params = [
                    {"params": self.model.get_1x_lr_params(), "lr": args.lr},
                    {"params": self.model.get_10x_lr_params(), "lr": args.lr * 10},
                ]

                optimizer = torch.optim.SGD(
                    train_params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov
                )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 - epoch / args.epochs)

        epoch_loss = []

        for epoch in range(args.epochs):
            t = time.time()
            batch_loss = []
            logging.info("Trainer_ID: {0}, Epoch: {1}".format(self.id, epoch))

            for (batch_idx, batch) in enumerate(train_data):
                x, labels, paths, _ = batch
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels).to(device)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                if batch_idx % 100 == 0:
                    logging.info(
                        "Trainer_ID: {0} Iteration: {1}, Loss: {2}, Time Elapsed: {3}".format(
                            self.id, batch_idx, loss, (time.time() - t) / 60
                        )
                    )

            scheduler.step()

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    "(Trainer_ID: {}. Local Training Epoch: {} \tLoss: {:.6f}".format(
                        self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                    )
                )

    def test(self, test_data, device):
        logging.info("Evaluating on Trainer ID: {}".format(self.id))
        model = self.model
        args = self.args

        model.eval()
        model.to(device)

        t = time.time()

        pass

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None):
        pass
