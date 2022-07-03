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

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        logging.info("----------test_on_the_server--------")

        model_list, micro_list, macro_list = [], [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            score, model = self.test(test_data, device)
            for idx in range(len(model_list)):
                self._compare_models(model, model_list[idx])
            model_list.append(model)
            micro_list.append(score)
            logging.info("Client {}, Test Micro F1 = {}".format(client_idx, score))
            if args.enable_wandb:
                wandb.log({"Client {} Test/Micro F1".format(client_idx): score})

        avg_micro = np.mean(np.array(micro_list))
        logging.info("Test Micro F1 = {}".format(avg_micro))
        if args.enable_wandb:
            wandb.log({"Test/ Micro F1": avg_micro})

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
