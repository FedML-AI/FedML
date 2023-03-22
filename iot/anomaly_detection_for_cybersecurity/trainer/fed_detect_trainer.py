import logging

import torch
from torch import nn

from fedml.core import ClientTrainer


class FedDetectTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, x in enumerate(train_data):
                x = x.to(device).float()
                optimizer.zero_grad()
                decode = model(x)
                loss = criterion(decode, x)
                loss.backward()
                optimizer.step()

                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(self.id, epoch, sum(epoch_loss) / len(epoch_loss))
            )

    def test(self, test_data, device, args):
        pass
