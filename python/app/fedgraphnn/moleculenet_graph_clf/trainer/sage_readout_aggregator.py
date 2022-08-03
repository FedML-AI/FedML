import logging

import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


# Trainer for MoleculeNet. The evaluation metric is ROC-AUC
from fedml.core import ServerAggregator


class SageMoleculeNetAggregator(ServerAggregator):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters, strict=False)


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
                forest = [level.to(device=device, dtype=torch.long, non_blocking=True) for level in forest]
                feature_matrix = feature_matrix.to(device=device, dtype=torch.float32, non_blocking=True)

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

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
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
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
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
