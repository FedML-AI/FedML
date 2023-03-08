import logging

import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from fedml.core import ClientTrainer


# Trainer for MoleculeNet. The evaluation metric is ROC-AUC


class SageMoleculeNetTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        test_data = None
        try:
            test_data = self.test_data
        except:
            pass

        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        max_test_score = 0
        best_model_params = {}
        for epoch in range(args.epochs):
            for mol_idxs, (adj_matrix, feature_matrix, label, mask) in enumerate(
                train_data
            ):
                # Pass on molecules that have no labels
                if torch.all(mask == 0).item():
                    continue

                optimizer.zero_grad()

                adj_matrix = adj_matrix.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                feature_matrix = feature_matrix.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                label = label.to(device=device, dtype=torch.float32, non_blocking=True)
                mask = mask.to(device=device, dtype=torch.float32, non_blocking=True)

                logits = model(adj_matrix, feature_matrix)
                loss = criterion(logits, label) * mask
                loss = loss.sum() / mask.sum()

                loss.backward()
                optimizer.step()

                if ((mol_idxs + 1) % args.frequency_of_the_test == 0) or (
                    mol_idxs == len(train_data) - 1
                ):
                    if test_data is not None:
                        test_score, _ = self.test(self.test_data, device, args)
                        print(
                            "Epoch = {}, Iter = {}/{}: Test Score = {}".format(
                                epoch, mol_idxs + 1, len(train_data), test_score
                            )
                        )
                        if test_score > max_test_score:
                            max_test_score = test_score
                            best_model_params = {
                                k: v.cpu() for k, v in model.state_dict().items()
                            }
                        print("Current best = {}".format(max_test_score))

        return max_test_score, best_model_params

    def test(self, test_data, device, args):
        logging.info("----------test--------")
        model = self.model
        model.eval()
        model.to(device)

        with torch.no_grad():
            y_pred = []
            y_true = []
            masks = []
            for mol_idx, (adj_matrix, feature_matrix, label, mask) in enumerate(
                test_data
            ):
                adj_matrix = adj_matrix.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                feature_matrix = feature_matrix.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )

                logits = model(adj_matrix, feature_matrix)

                y_pred.append(logits.cpu().numpy())
                y_true.append(label.cpu().numpy())
                masks.append(mask.numpy())

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        masks = np.array(masks)

        results = []
        for label in range(masks.shape[1]):
            valid_idxs = np.nonzero(masks[:, label])
            truth = y_true[valid_idxs, label].flatten()
            pred = y_pred[valid_idxs, label].flatten()

            if np.all(truth == 0.0) or np.all(truth == 1.0):
                results.append(float("nan"))
            else:
                if args.metric == "prc-auc":
                    precision, recall, _ = precision_recall_curve(truth, pred)
                    score = auc(recall, precision)
                else:
                    score = roc_auc_score(truth, pred)

                results.append(score)

        score = np.nanmean(results)

        return score, model
