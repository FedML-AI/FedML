import logging

import torch
from torch import nn

from .utils import transform_tensor_to_list


class FedAvgRobustTrainer(object):
    def __init__(
        self,
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        train_data_num,
        device,
        model,
        poisoned_train_loader,
        num_dps_poisoned_dataset,
        args,
    ):
        # TODO(@hwang595): double check if this makes sense with Chaoyang
        # here we always assume the client with `client_index=1` as the attacker
        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.all_train_data_num = train_data_num

        self.poisoned_train_loader = poisoned_train_loader
        self.num_dps_poisoned_dataset = num_dps_poisoned_dataset

        if self.client_index == 1:
            self.train_local = self.poisoned_train_loader
            self.local_sample_number = self.num_dps_poisoned_dataset
        else:
            self.train_local = self.train_data_local_dict[client_index]
            self.local_sample_number = self.train_data_local_num_dict[client_index]

        self.device = device
        self.args = args
        self.model = model
        # logging.info(self.model)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # TODO(hwang): since we only added the black-box attack now, we assume that the attacker uses the same hyper-params with the honest clients
        if self.args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.args.learning_rate
            )
        else:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.args.learning_rate,
                weight_decay=self.args.wd,
                amsgrad=True,
            )

    def update_model(self, weights):
        # logging.info("update_model. client_index = %d" % self.client_index)
        self.model.load_state_dict(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        if (
            self.client_index == 1
        ):  # TODO(@hwang595): double check if this makes sense with Chaoyang, we make it the attacker
            self.train_local = self.poisoned_train_loader
            self.local_sample_number = self.num_dps_poisoned_dataset
        else:
            self.train_local = self.train_data_local_dict[client_index]
            self.local_sample_number = self.train_data_local_num_dict[client_index]

    def train(self):
        self.model.to(self.device)
        # change to train mode
        self.model.train()

        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_local):
                # logging.info(images.shape)
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                if len(batch_loss) > 0:
                    epoch_loss.append(sum(batch_loss) / len(batch_loss))
                    logging.info(
                        "(client {}. Local Training Epoch: {} \tLoss: {:.6f}".format(
                            self.client_index, epoch, sum(epoch_loss) / len(epoch_loss)
                        )
                    )

        weights = self.model.cpu().state_dict()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number
