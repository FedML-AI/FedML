import copy
import logging
import time

import numpy as np
import torch
from flamby.datasets.fed_kits19 import (
    LR,
    softmax_helper,
)
from torch.optim import lr_scheduler

from fedml.core.alg_frame.client_trainer import ClientTrainer


class KITS19Trainer(ClientTrainer):
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

        from flamby.datasets.fed_kits19.loss import BaselineLoss

        loss_func = BaselineLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), LR, weight_decay=3e-5, amsgrad=True
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=30,
            verbose=True,
            threshold=1e-3,
            threshold_mode="abs",
        )

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        model = model.to(device)
        # To draw loss and accuracy plots
        training_loss_list = []
        training_dice_list = []
        logging.info(" Train Data Size " + str(len(train_data.dataset)))
        # logging.info(" Test Data Size " + str(dataset_sizes["test"]))
        for epoch in range(epochs):
            logging.info("Epoch {}/{}".format(epoch, epochs - 1))
            logging.info("-" * 10)

            dice_list = []
            running_loss = 0.0
            dice_score = 0.0
            model.train()  # Set model to training mode

            # Iterate over data.
            for sample in train_data:
                inputs = sample[0].to(device)
                labels = sample[1].to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = loss_func(outputs, labels)

                    # backward + optimize only if in training phase, record training loss
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * inputs.size(0)

                # if phase == "train":
                #     scheduler.step(epoch)

            epoch_loss = running_loss / len(train_data.dataset)
            epoch_acc = np.mean(dice_list)  # average dice

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            logging.info(
                "Training Loss: {:.4f} Validation Acc: {:.4f} ".format(
                    epoch_loss, epoch_acc
                )
            )
            training_loss_list.append(epoch_loss)
            training_dice_list.append(epoch_acc)

        time_elapsed = time.time() - since
        logging.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        logging.info("Best test Balanced acc: {:4f}".format(best_acc))
        logging.info("----- Training Loss ---------")
        logging.info(training_loss_list)
        logging.info("------Validation Accuracy ------")
        logging.info(training_dice_list)
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
