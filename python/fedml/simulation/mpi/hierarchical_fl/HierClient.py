import copy

import torch
import torch.nn as nn

from ...sp.fedavg.client import Client


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

    def train(self, w, scaled_loss_factor=1.0):
        self.model.load_state_dict(w)
        self.model.to(self.device)

        scaled_loss_factor = min(scaled_loss_factor, 1.0)
        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate * scaled_loss_factor)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args.learning_rate * scaled_loss_factor,
                weight_decay=self.args.weight_decay,
                amsgrad=True,
            )

        for epoch in range(self.args.epochs):
            for x, labels in self.local_training_data:
                x, labels = x.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                log_probs = self.model(x)
                loss = self.criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()
                optimizer.step()

        return copy.deepcopy(self.model.cpu().state_dict())
