import copy
import logging
import time

import numpy as np
import torch

from flamby.datasets.fed_tcga_brca import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedTcgaBrca,
    metric,
)
from flamby.utils import evaluate_model_on_tests
from torch.optim import lr_scheduler

from fedml.core.alg_frame.client_trainer import ClientTrainer


class TcgaBrcaTrainer(ClientTrainer):
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

        from flamby.datasets.fed_tcga_brca import BaselineLoss

        loss_func = BaselineLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        model = model.to(device)
        # To draw loss and accuracy plots
        training_loss_list = []
        training_cindex_list = []

        y_true_list = []
        y_pred_list = []

        print(" Train Data Size " + str(len(train_data.dataset)))
        # print(" Test Data Size " + str(dataset_sizes["test"]))
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch, epochs - 1))
            print("-" * 10)

            running_loss = 0.0
            c_index = 0.0
            model.train()  # Set model to training mode

            # Iterate over data.
            for idx, (X, y) in enumerate(train_data):
                X = X.to(device)
                y = y.to(device)
                y_true_list.append(y)

                optimizer.zero_grad()
                y_pred = model(X)
                y_pred_list.append(y_pred)
                loss = loss_func(y_pred, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                #print(y[:,1], -y_pred, y[:,0])
                #c_index += metric(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

            epoch_loss = running_loss / len(train_data.dataset)
            y_true = torch.cat(y_true_list)
            y_hat = torch.cat(y_pred_list)
            epoch_c_index = metric(y_true.cpu().detach().numpy(), y_hat.cpu().detach().numpy())
            #epoch_c_index = c_index / len(train_data.dataset)

            print(
                "Training Loss: {:.4f} Validation C-Index: {:.4f} ".format(
                    epoch_loss, epoch_c_index
                )
            )
            training_loss_list.append(epoch_loss)
            training_cindex_list.append(epoch_c_index)

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("----- Training Loss ---------")
        print(training_loss_list)
        print("------Validation C-Index ------")
        print(training_cindex_list)
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
