import logging

import numpy as np
import torch
import wandb
from sklearn.metrics import confusion_matrix

from fedml.core.alg_frame.client_trainer import ClientTrainer


# Trainer for MoleculeNet. The evaluation metric is ROC-AUC


class FedNodeClfTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )

        max_test_score, max_val_score = 0, 0
        best_model_params = {}
        for epoch in range(args.epochs):
            for idx_batch, batch in enumerate(train_data):
                batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                label = batch.y
                loss = model.loss(pred, label)
                loss.backward()
                optimizer.step()

        return max_test_score, best_model_params

    def test(self, test_data, device):
        logging.info("----------test--------")
        model = self.model
        model.eval()
        model.to(device)
        conf_mat = np.zeros((self.model.nclass, self.model.nclass))

        with torch.no_grad():
            for batch_index, batch in enumerate(test_data):
                # logging.info("batch_index = {}. batch = {}.".format(batch_index, batch))
                batch.to(device)

                pred = model(batch)
                label = batch.y
                cm_result = confusion_matrix(
                    label.cpu().numpy().flatten(),
                    pred.argmax(dim=1).cpu().numpy().flatten(),
                    labels=np.arange(0, self.model.nclass),
                )
                conf_mat += cm_result

        # logging.info("conf_mat = {}".format(conf_mat))

        # Compute Micro F1
        TP = np.trace(conf_mat)
        # logging.info("TP = {}".format(TP))
        FP = np.sum(conf_mat) - TP
        # logging.info("FP = {}".format(FP))
        FN = FP
        micro_pr = TP / (TP + FP)
        micro_rec = TP / (TP + FN)
        # logging.info("micro_pr = {}, micro_rec = {}".format(micro_pr, micro_rec))
        if micro_pr + micro_rec == 0.0:
            denominator = micro_pr + micro_rec + np.finfo(float).eps
        else:
            denominator = micro_pr + micro_rec
        micro_F1 = 2 * micro_pr * micro_rec / denominator
        logging.info("score = {}".format(micro_F1))
        return micro_F1, model
