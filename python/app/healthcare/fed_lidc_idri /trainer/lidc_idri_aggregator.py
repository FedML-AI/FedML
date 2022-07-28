import logging

import numpy as np
import torch
import torch.nn as nn

import fedml
from fedml.core import ServerAggregator


class LIDCAggregator(ServerAggregator):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def test(self, test_data, device, args):
        pass

    def _test(self, test_data, device):
        ## TODO: nchunks pass as an argument
        nchunks = 9
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

        from flamby.datasets.fed_lidc_idri.metric import metric

        with torch.inference_mode():
            dice_list = []
            for (X, y) in test_data:
                intersection = 0
                union = 0
                X = torch.chunk(X, nchunks)
                y = torch.chunk(y, nchunks)
                for ii, X_ in enumerate(X):
                    y_ = y[ii]
                    if torch.cuda.is_available() and use_gpu:
                        X_ = X_.cuda()
                        model = model.cuda()
                    y_pred = model(X_).detach().cpu().numpy()
                    y_ = y_.detach().cpu().numpy()
                    intersection += np.sum(y_pred * y_)
                    union += np.sum(0.5 * (y_pred + y_))
                dice = 1 if np.abs(union) < 1e-7 else intersection / union
                dice_list.append(dice)
            test_metrics = np.mean(dice_list)

        logging.info(f"Test metrics: {test_metrics}")
        return test_metrics

    def test_all(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        return True
