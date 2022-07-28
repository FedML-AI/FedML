import copy
import logging
import time

import numpy as np
import torch
from sklearn import metrics
from flamby.datasets.fed_isic2019 import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedIsic2019,
    metric,
)
from flamby.utils import evaluate_model_on_tests
from torch.optim import lr_scheduler

from fedml.core.alg_frame.client_trainer import ClientTrainer


class ISIC2019Trainer(ClientTrainer):
    def __init__(self, model, args=None):
        super().__init__(model, args)

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        logging.info("Start training on Trainer {}".format(self.id))
        model = self.model
        args = self.args

        epochs = args.epochs  # number of epochs

        loss_func = BaselineLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[3, 5, 7, 9, 11, 13, 15, 17], gamma=0.5
        )

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        model = model.to(device)
        # To draw loss and accuracy plots
        training_loss_list = []
        training_acc_list = []

        logging.info(" Train Data Size " + str(len(train_data.dataset)))
        # logging.info(" Test Data Size " + str(dataset_sizes["test"]))
        for epoch in range(epochs):
            logging.info("Epoch {}/{}".format(epoch, epochs - 1))
            logging.info("-" * 10)

            running_loss = 0.0
            running_acc = 0.0
            model.train()  # Set model to training mode
            y_pred = []
            y_true = []

            # Iterate over data.
            for idx, (X, y) in enumerate(train_data):
                y_true.append(y)
                X = X.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                outputs = model(X)
                loss = loss_func(outputs, y)
                _, preds = torch.max(outputs, 1)
                y_pred.append(preds)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X.size(0)
                running_acc += torch.sum(preds == y.data)

                if idx % 10 == 0:
                    logging.info(
                        "Iter: {}, Loss: {}, Acc: {}".format(
                            idx,
                            loss.item(),
                            running_acc.double() / (idx + 1) / args.batch_size,
                        )
                    )

            scheduler.step()

            epoch_loss = running_loss / len(train_data.dataset)
            epoch_acc = running_acc.double() / len(train_data.dataset)
            y = torch.cat(y_true)
            y_hat = torch.cat(y_pred)

            epoch_balanced_acc = metrics.balanced_accuracy_score(y.cpu(), y_hat.cpu())

            logging.info(
                "{} Loss: {:.4f} Acc: {:.4f} Balanced acc: {:.4f}".format(
                    "train", epoch_loss, epoch_acc, epoch_balanced_acc
                )
            )
            training_loss_list.append(epoch_loss)
            training_acc_list.append(epoch_acc)

        time_elapsed = time.time() - since
        logging.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        logging.info("----- Training Loss ---------")
        logging.info(training_loss_list)
        logging.info("------Validation AUC ------")
        logging.info(training_acc_list)
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
