import logging

import numpy as np
import torch
import torch.nn as nn

import fedml
from fedml.core import ServerAggregator


class TcgaBrcaAggregator(ServerAggregator):
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

        from flamby.datasets.fed_tcga_brca.metric import metric

        y_true_list = []
        y_pred_list = []

        with torch.inference_mode():
            c_index_list = []
            for (X, y) in test_data:
                X, y = X.to(device), y.to(device)
                y_true_list.append(y)
                y_pred = model(X)
                y_pred_list.append(y_pred)
            
            y_true = torch.cat(y_true_list)
            y_hat = torch.cat(y_pred_list)
            c_index = metric(y_true.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
            c_index_list.append(c_index)

            test_metrics = np.mean(c_index_list)

        logging.info(f"Test metrics: {test_metrics}")
        return test_metrics

    def test_all(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        return True
