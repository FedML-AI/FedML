import logging
import time

import torch
from torch import nn


class FedAvgRobustTrainer(object):
    def __init__(self, client_index, train_local, local_sample_number, all_train_data_num, device, model,
                 args):
        self.client_index = client_index
        self.train_local = train_local
        self.local_sample_number = local_sample_number
        self.all_train_data_num = all_train_data_num
        self.device = device
        self.args = args
        self.model = model
        # logging.info(self.model)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.5)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
                                          weight_decay=0.0001, amsgrad=True)

    def update_model(self, weights):
        logging.info("update_model. client_index = %d" % self.client_index)
        self.model.load_state_dict(weights)

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
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.client_index,
                                                                            epoch, sum(epoch_loss) / len(epoch_loss)))

        start_time = time.time()
        weights = self.model.cpu().state_dict()
        end_time = time.time()
        logging.info("CPU to GPU cost = %d" % (end_time - start_time))

        return weights, self.local_sample_number

    def infer(self):
        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        test_data = self.train_local
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)

                pred = self.model(x)
                loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
            logging.info("client_idx = %d, local_train_loss = %s" % (self.client_index, test_loss))
        return test_correct / test_sample_number, test_loss
