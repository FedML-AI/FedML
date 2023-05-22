import logging

import numpy as np
import torch
import torch.nn as nn

import fedml
from fedml.core import ServerAggregator


class HeartDiseaseAggregator(ServerAggregator):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def test(self, test_data, device, args):
        pass

    def _test(self, test_data, device):
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

    def test_all(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        model = self.model
        args = self.args
        from flamby.datasets.fed_heart_disease.metric import metric
        from flamby.datasets.fed_heart_disease import BaselineLoss

        test_metrics = {
            "test_correct": 0,
            "test_total": 0,
            "test_loss": 0,
        }
        
        if not test_data_local_dict:
            logging.info("No test data for this trainer")
            return test_metrics

        model.eval()
        model.to(device)

        to_avg_auc_list, to_avg_loss_list = [], []
        w = []
        debug_list = []
        for i in range(args.client_num_per_round):
            loss_func = BaselineLoss()
            with torch.inference_mode():
                auc_list = []
                loss_list = []
                w.append(len(test_data_local_dict[i]))
                for (X, y) in test_data_local_dict[i]:
                    X, y = X.to(device), y.to(device)
                    y_pred = model(X)

                    if len(debug_list) == 0:
                        debug_list.append(y_pred.detach().cpu().numpy())

                    auc = metric(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
                    loss = loss_func(y_pred, y)

                    auc_list.append(auc)
                    loss_list.append(loss.item())
                
                test_auc_metrics = np.mean(auc_list)
                test_loss_metrics = np.mean(loss_list)
            to_avg_auc_list.append(test_auc_metrics)
            to_avg_loss_list.append(test_loss_metrics)
        # avg
        avg_auc = np.average(to_avg_auc_list, weights=w)
        avg_loss = np.average(to_avg_loss_list, weights=w)
        print("Average AUC performance", avg_auc)
        print("Average Loss performance", avg_loss)

        print ({"round_idx": args.round_idx, "loss": avg_loss, "evaluation_result": avg_auc})
        fedml.mlops.log({"round_idx": args.round_idx, "loss": avg_loss, "evaluation_result": avg_auc})
        test_metrics["test_correct"] = avg_auc
        test_metrics["test_total"] = avg_auc
        test_metrics["test_loss"] = avg_loss
        return test_metrics
