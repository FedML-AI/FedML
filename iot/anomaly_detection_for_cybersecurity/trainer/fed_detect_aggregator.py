import logging
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from fedml.core import ServerAggregator


class FedDetectAggregator(ServerAggregator):

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def test(self, test_data, device, args):
        pass

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        model = self.model

        model.to(device)
        model.eval()

        true_negative = 0
        false_positive = 0
        true_positive = 0
        false_negative = 0

        threshold = self._get_threshold_global(args, device)
        threshold_func = nn.MSELoss(reduction="none")

        for client_index in train_data_local_dict.keys():
            train_data = train_data_local_dict[client_index]
            for batch_idx, x in enumerate(train_data):
                x = x.to(device).float()
                diff = threshold_func(model(x), x)
                mse = diff.mean(dim=1)
                false_positive += sum(mse > threshold)
                true_negative += sum(mse <= threshold)

        for client_index in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_index]
            for batch_idx, x in enumerate(test_data):
                x = x.to(device).float()
                diff = threshold_func(model(x), x)
                mse = diff.mean(dim=1)
                true_positive += sum(mse > threshold)
                false_negative += sum(mse <= threshold)

        accuracy = (true_positive + true_negative) / (
            true_positive + true_negative + false_positive + false_negative
        )
        precision = true_positive / (true_positive + false_positive)
        false_positive_rate = false_positive / (false_positive + true_negative)
        tpr = true_positive / (true_positive + false_negative)
        tnr = true_negative / (true_negative + false_positive)

        logging.info("The True negative number is {}".format(true_negative))
        logging.info("The False positive number is {}".format(false_positive))
        logging.info("The True positive number is {}".format(true_positive))
        logging.info("The False negative number is {}".format(false_negative))

        logging.info("The accuracy is {}".format(accuracy))
        logging.info("The precision is {}".format(precision))
        logging.info("The false positive rate is {}".format(false_positive_rate))
        logging.info("tpr is {}".format(tpr))
        logging.info("tnr is {}".format(tnr))

        return True


    def _get_threshold_global(self, args, device):
        device_list = [
            "Danmini_Doorbell",
            "Ecobee_Thermostat",
            "Ennio_Doorbell",
            "Philips_B120N10_Baby_Monitor",
            "Provision_PT_737E_Security_Camera",
            "Provision_PT_838_Security_Camera",
            "Samsung_SNH_1011_N_Webcam",
            "SimpleHome_XCS7_1002_WHT_Security_Camera",
            "SimpleHome_XCS7_1003_WHT_Security_Camera",
        ]
        th_local_dict = dict()
        min_max_file_path = "./data"
        min_dataset = np.loadtxt(os.path.join(min_max_file_path, "min_dataset.txt"))
        max_dataset = np.loadtxt(os.path.join(min_max_file_path, "max_dataset.txt"))
        for i, device_name in enumerate(device_list):
            benign_data = pd.read_csv(
                os.path.join(args.data_cache_dir, device_name, "benign_traffic.csv")
            )
            benign_data = np.array(benign_data)

            benign_th = benign_data[5000:8000]
            benign_th[np.isnan(benign_th)] = 0
            benign_th = (benign_th - min_dataset) / (max_dataset - min_dataset)

            th_local_dict[i] = torch.utils.data.DataLoader(
                benign_th, batch_size=128, shuffle=False, num_workers=0
            )

        model = self.model
        model.to(device)
        model.eval()

        mse = list()
        threshold_func = nn.MSELoss(reduction="none")
        for client_index in th_local_dict.keys():
            train_data = th_local_dict[client_index]
            for batch_idx, x in enumerate(train_data):
                x = x.to(device).float()
                diff = threshold_func(model(x), x)
                mse.append(diff)

        mse_global = torch.cat(mse).mean(dim=1)
        threshold_global = torch.mean(mse_global) + 3 * torch.std(mse_global)

        return threshold_global