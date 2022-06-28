import logging

import numpy as np
import torch
import wandb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from fedml.core.alg_frame.client_trainer import ClientTrainer


# Trainer for MoleculeNet. The evaluation metric is RMSE, by default


class GcnMoleculeNetTrainer(ClientTrainer):
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

        criterion = torch.nn.MSELoss() if args.dataset != "qm9" else torch.nn.L1Loss()
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
            for mol_idxs, (adj_matrix, feature_matrix, label, _) in enumerate(
                train_data
            ):
                optimizer.zero_grad()

                adj_matrix = adj_matrix.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                feature_matrix = feature_matrix.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                label = label.to(device=device, dtype=torch.float32, non_blocking=True)

                logits = model(adj_matrix, feature_matrix)
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
            for mol_idx, (adj_matrix, feature_matrix, label, _) in enumerate(test_data):
                adj_matrix = adj_matrix.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                feature_matrix = feature_matrix.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                label = label.to(device=device, dtype=torch.float32, non_blocking=True)
                logits = model(adj_matrix, feature_matrix)
                y_pred.append(logits.cpu().numpy())
                y_true.append(label.cpu().numpy())

            if args.metric == "rmse":
                score = mean_squared_error(
                    np.array(y_true), np.array(y_pred), squared=False
                )
            elif args.metric == "r2":
                score = r2_score(np.array(y_true), np.array(y_pred))
            else:
                score = mean_absolute_error(np.array(y_true), np.array(y_pred))
        return score, model

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        logging.info("----------test_on_the_server--------")
        # for client_idx in train_data_local_dict.keys():
        #     train_data = train_data_local_dict[client_idx]
        #     train_score = self.test(train_data, device, args)
        #     logging.info('Client {}, Train ROC-AUC score = {}'.format(client_idx, train_score))

        model_list, score_list = [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            score, model = self.test(test_data, device, args)
            for idx in range(len(model_list)):
                self._compare_models(model, model_list[idx])
            model_list.append(model)
            score_list.append(score)
            logging.info(
                "Client {}, Test {} = {}".format(client_idx, args.metric.upper(), score)
            )
            if args.enable_wandb:
                wandb.log(
                    {"Client {} Test/{}".format(client_idx, args.metric.upper()): score}
                )
        avg_score = np.mean(np.array(score_list))
        logging.info("Test {} score = {}".format(args.metric.upper(), avg_score))
        if args.enable_wandb:
            wandb.log({"Test/{}}".format(args.metric.upper()): avg_score})

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
                    logging.info("Mismatch found at", key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            logging.info("Models match perfectly! :)")
