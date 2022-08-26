import copy

import torch
import torch.nn as nn

from ..fedavg.client import Client


class HFLClient(Client):
    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model,
                 model_trainer):

        super().__init__(client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                         model_trainer)
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

        self.args = args
        self.device = device
        self.model = model
        self.model_trainer = model_trainer
        self.criterion = nn.CrossEntropyLoss().to(device)

    def train(self, global_round_idx, group_round_idx, w):
        self.model.load_state_dict(w)
        self.model.to(self.device)

        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args.lr,
                weight_decay=self.args.wd,
                amsgrad=True,
            )

        w_list = []
        for epoch in range(self.args.epochs):
            for x, labels in self.local_training_data:
                x, labels = x.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                log_probs = self.model(x)
                loss = self.criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()
                optimizer.step()
            global_epoch = (
                global_round_idx * self.args.group_comm_round * self.args.epochs
                + group_round_idx * self.args.epochs
                + epoch
            )
            if global_epoch % self.args.frequency_of_the_test == 0 or epoch == self.args.epochs - 1:
                w_list.append((global_epoch, copy.deepcopy(self.model.state_dict())))
        return w_list
