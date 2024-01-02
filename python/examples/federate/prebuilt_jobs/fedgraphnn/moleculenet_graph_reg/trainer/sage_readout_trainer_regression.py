import logging

import numpy as np
import torch
import wandb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from fedml.core.alg_frame.client_trainer import ClientTrainer


# Trainer for MoleculeNet. The evaluation metrics are RMSE, R2, and MAE


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

        criterion = torch.nn.MSELoss() if args.dataset != "qm9" else torch.nn.MAELoss()
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        min_score = np.Inf if args.metric != "r2" else -np.Inf

        best_model_params = {}
        # print('Training on {}'.format(torch.cuda.get_device_name()))
        for epoch in range(args.epochs):
            avg_loss = 0
            count = 0
            for mol_idxs, (forest, feature_matrix, label, _) in enumerate(train_data):
                optimizer.zero_grad()

                forest = [
                    level.to(device=device, dtype=torch.long, non_blocking=True)
                    for level in forest
                ]
                feature_matrix = feature_matrix.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                label = label.to(device=device, dtype=torch.float32, non_blocking=True)

                logits = model(forest, feature_matrix)
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()

                if test_data is not None:
                    test_score, _ = self.test(self.test_data, device, args)
                    if args.metric != "r2":
                        print(
                            "Epoch = {}: Test {} = {}".format(
                                epoch, args.metric.upper(), test_score
                            )
                        )
                        if test_score < min_score:
                            min_score = test_score
                            best_model_params = {
                                k: v.cpu() for k, v in model.state_dict().items()
                            }
                        print(
                            "Current best {}= {}".format(args.metric.upper(), min_score)
                        )
                    else:
                        print("Epoch = {}: Test R2 = {}".format(epoch, test_score))
                        if test_score > min_score:
                            min_score = test_score
                            best_model_params = {
                                k: v.cpu() for k, v in model.state_dict().items()
                            }
                        print("Current best R2= {}".format(min_score))
            #
            #     avg_loss += loss.item()
            #     count += 1
            #     # logging.info("training. epoch = %d, mol_idxs = %d, avg_loss = %f" % (epoch, mol_idxs, avg_loss))
            #
            # avg_loss /= count
        return min_score, best_model_params

    def test(self, test_data, device, args):
        logging.info("----------test--------")
        model = self.model
        model.eval()
        model.to(device)

        with torch.no_grad():
            y_pred = []
            y_true = []
            for mol_idx, (forest, feature_matrix, label, _) in enumerate(test_data):
                forest = [
                    level.to(device=device, dtype=torch.long, non_blocking=True)
                    for level in forest
                ]
                feature_matrix = feature_matrix.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                label = label.to(device=device, dtype=torch.float32, non_blocking=True)
                logits = model(forest, feature_matrix)
                y_pred.append(logits.cpu().numpy())
                y_true.append(label.cpu().numpy())

            # logging.info(y_true)
            # logging.info(y_pred)
            if args.metric == "rmse":
                score = mean_squared_error(
                    np.array(y_true), np.array(y_pred), squared=False
                )
            elif args.metric == "r2":
                score = r2_score(np.array(y_true), np.array(y_pred))
            else:
                score = mean_absolute_error(np.array(y_true), np.array(y_pred))

        return score, model
