import copy
import logging
import time

import numpy as np
import torch

from flamby.datasets.fed_heart_disease import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedHeartDisease,
    metric,
)
from flamby.utils import evaluate_model_on_tests
from torch.optim import lr_scheduler

from fedml.core.alg_frame.client_trainer import ClientTrainer


class HeartDiseaseTrainer(ClientTrainer):
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

        from flamby.datasets.fed_heart_disease import BaselineLoss

        loss_func = BaselineLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        model = model.to(device)
        # To draw loss and accuracy plots
        training_loss_list = []
        training_auc_list = []

        print(" Train Data Size " + str(len(train_data.dataset)))
        # print(" Test Data Size " + str(dataset_sizes["test"]))
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch, epochs - 1))
            print("-" * 10)

            running_loss = 0.0
            auc = 0.0
            model.train()  # Set model to training mode

            # Iterate over data.
            for idx, (X, y) in enumerate(train_data):
                X = X.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                y_pred = model(X)
                loss = loss_func(y_pred, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                auc += metric(y_pred.detach().cpu().numpy(), y.detach().cpu().numpy())

            epoch_loss = running_loss / len(train_data.dataset)
            epoch_auc = auc / len(train_data.dataset)

            print(
                "Training Loss: {:.4f} Validation AUC: {:.4f} ".format(
                    epoch_loss, epoch_auc
                )
            )
            training_loss_list.append(epoch_loss)
            training_auc_list.append(epoch_auc)

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("----- Training Loss ---------")
        print(training_loss_list)
        print("------Validation AUC ------")
        print(training_auc_list)
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

        from flamby.datasets.fed_heart_disease.metric import metric

        with torch.inference_mode():
            auc_list = []
            for (X, y) in test_data:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                auc = metric(y_pred.detach().cpu().numpy(), y.detach().cpu().numpy())
                auc_list.append(auc)

            test_metrics = np.mean(auc_list)

        logging.info(f"Test metrics: {test_metrics}")
        return test_metrics

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ):
        return True
