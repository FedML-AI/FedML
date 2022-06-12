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
            for mol_idxs, (forest, feature_matrix, label, mask) in enumerate(
                train_data
            ):
                # Pass on molecules that have no labels
                if torch.all(mask == 0).item():
                    continue

                optimizer.zero_grad()

                forest = [
                    level.to(device=device, dtype=torch.long, non_blocking=True)
                    for level in forest
                ]
                sizes = [level.size() for level in forest]
                types = [level.dtype for level in forest]

                feature_matrix = feature_matrix.to(device=device, dtype=torch.float32, non_blocking=True)
                label = label.to(device=device, dtype=torch.float32, non_blocking=True)
                mask = mask.to(device=device, dtype=torch.float32, non_blocking=True)

                logits = model(forest, feature_matrix)
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
            for mol_idx, (forest, feature_matrix, label, mask) in enumerate(test_data):
                forest = [
                    level.to(device=device, dtype=torch.long, non_blocking=True)
                    for level in forest
                ]
                feature_matrix = feature_matrix.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )

                logits = model(forest, feature_matrix)

                y_pred.append(logits.cpu().numpy())
                y_true.append(label.numpy())
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

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        logging.info("----------test_on_the_server--------")

        model_list, score_list = [], []
        for client_idx in test_data_local_dict.keys():

            test_data = test_data_local_dict[client_idx]
            score, model = self.test(test_data, device, args)
            for idx in range(len(model_list)):
                self._compare_models(model, model_list[idx])
            model_list.append(model)
            score_list.append(score)
            logging.info("Client {}, Test ROC-AUC score = {}".format(client_idx, score))
            if args.enable_wandb:
                wandb.log({"Client {} Test/ROC-AUC".format(client_idx): score})
        avg_score = np.mean(np.array(score_list))
        logging.info("Test ROC-AUC Score = {}".format(avg_score))
        if args.enable_wandb:
            wandb.log({"Test/ROC-AUC": avg_score})
        return True

    def _compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(
            model_1.state_dict().items(), model_2.state_dict().items()
        ):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if key_item_1[0] == key_item_2[0]:
                    logging.info("Mismtach found at", key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            logging.info("Models match perfectly! :)")
