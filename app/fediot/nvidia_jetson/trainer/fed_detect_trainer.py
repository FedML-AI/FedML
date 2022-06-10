import torch
from torch import nn
import numpy as np
import pandas as pd
import os

from fedml.core.alg_frame.client_trainer import ClientTrainer
import logging


class MyModelTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                logging.info(
                    "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        (batch_idx + 1) * args.batch_size,
                        len(train_data) * args.batch_size,
                        100.0 * (batch_idx + 1) / len(train_data),
                        loss.item(),
                    )
                )
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )

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

        threshold = self.get_threshold_global(args)
        threshold_func = nn.MSELoss(reduction="none")

        for client_index in train_data_local_dict.keys():
            train_data = train_data_local_dict[client_index]
            for idx, inp in enumerate(train_data):
                inp = inp.to(device)
                diff = threshold_func(model(inp), inp)
                mse = diff.mean(dim=1)
                false_positive += sum(mse > threshold)
                true_negative += sum(mse <= threshold)

        for client_index in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_index]
            for idx, inp in enumerate(test_data):
                inp = inp.to(device)
                diff = threshold_func(model(inp), inp)
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

        print("The True negative number is ", true_negative)
        print("The False positive number is ", false_positive)
        print("The True positive number is ", true_positive)
        print("The False negative number is ", false_negative)

        print("The accuracy is ", accuracy)
        print("The precision is ", precision)
        print("The false positive rate is ", false_positive_rate)
        print("tpr is ", tpr)
        print("tnr is ", tnr)

        return True

    def get_threshold_global(self, args):
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
        min_dataset = np.loadtxt(os.path.join(args.data_cache_dir, "min_dataset.txt"))
        max_dataset = np.loadtxt(os.path.join(args.data_cache_dir, "max_dataset.txt"))
        for i, device in enumerate(device_list):
            benign_data = pd.read_csv(
                os.path.join(args.data_cache_dir, device, "benign_traffic.csv")
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
            for idx, inp in enumerate(train_data):
                inp = inp.to(device)
                diff = threshold_func(model(inp), inp)
                mse.append(diff)

        mse_global = torch.cat(mse).mean(dim=1)
        threshold_global = torch.mean(mse_global) + 1 * torch.std(mse_global)

        return threshold_global
