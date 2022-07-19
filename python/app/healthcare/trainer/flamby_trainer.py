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
import argparse
import copy
import os
import time

import numpy as np
import torch
from torch.optim import lr_scheduler

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

from fedml.core.alg_frame.client_trainer import ClientTrainer


class FlambyTrainer(ClientTrainer):
    def __init__(self, model, args=None):
        super().__init__(model, args)

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args=None):
        logging.info("Start training on Trainer {}".format(self.id))
        model = self.model
        args = self.args

        epochs = args.epochs  # number of epochs
        


        # def train_model(
        #     model, optimizer, scheduler, dataloaders, dataset_sizes, device, lossfunc, num_epochs
        # ):
        """Training function
        Parameters
        ----------
        model : torch model to be trained
        optimizer : torch optimizer used for training
        scheduler : torch scheduler used for training
        dataloaders : dictionary {"train": train_dataloader, "test": test_dataloader}
        dataset_sizes : dictionary {"train": len(train_dataset), "test": len(test_dataset)}
        device : device where model parameters are stored
        lossfunc : function, loss function
        num_epochs : int, numuber of epochs for training
        Returns
        -------
        model : torch model that scored the best test accuracy
        """

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        model = model.to(device)
        # To draw loss and accuracy plots
        training_loss_list = []
        training_dice_list = []
        print(" Train Data Size " + str(dataset_sizes["train"]))
        print(" Test Data Size " + str(dataset_sizes["test"]))
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)

            dice_list = []
            running_loss = 0.0
            dice_score = 0.0
            # Each epoch has a training and validation phase
            for phase in ["train", "test"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                # Iterate over data.
                for sample in dataloaders[phase]:
                    inputs = sample[0].to(device)
                    labels = sample[1].to(device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss = lossfunc(outputs, labels)

                        # backward + optimize only if in training phase, record training loss
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item() * inputs.size(0)

                        # if test: record dice
                        if phase == "test":
                            preds_softmax = softmax_helper(outputs)
                            preds = preds_softmax.argmax(1)
                            dice_score = metric(preds.cpu(), labels.cpu())
                            dice_list.append(dice_score)

                # if phase == "train":
                #     scheduler.step(epoch)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = np.mean(dice_list)  # average dice

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            print(
                "Training Loss: {:.4f} Validation Acc: {:.4f} ".format(epoch_loss, epoch_acc)
            )
            training_loss_list.append(epoch_loss)
            training_dice_list.append(epoch_acc)

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best test Balanced acc: {:4f}".format(best_acc))
        print("----- Training Loss ---------")
        print(training_loss_list)
        print("------Validation Accuracy ------")
        print(training_dice_list)
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def test(self, test_data, device, args):
        logging.info("Evaluating on Trainer ID: {}".format(self.id))
        model = self.model
        args = self.args

        test_metrics = {
            "test_correct": 0,
            "test_total": 0,
            "test_loss": 0,
        }

        if not test_data:
            logging.info("No test data for this trainer")
            return test_metrics

        model.eval()
        model.to(device)

        logging.info(f"Test metrics: {test_metrics}")
        return test_metrics

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ):
        return False
        