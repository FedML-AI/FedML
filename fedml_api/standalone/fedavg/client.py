import logging

import torch
from torch import nn


class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model = model

        '''
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        '''
        if self.args.dataset == "stackoverflow_lr":
            self.criterion = nn.BCELoss(reduction = 'sum').to(device)
        else:
            self.criterion = nn.CrossEntropyLoss().to(device)

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global):
        self.model.load_state_dict(w_global)
        self.model.to(self.device)

        # train and update
        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)

        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(self.local_training_data):
                x, labels = x.to(self.device), labels.to(self.device)
                # logging.info("x.size = " + str(x.size()))
                # logging.info("labels.size = " + str(labels.size()))
                self.model.zero_grad()
                log_probs = self.model(x)
                loss = self.criterion(log_probs, labels)
                loss.backward()

                # to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
                #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            #     self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))
        return self.model.cpu().state_dict(), sum(epoch_loss) / len(epoch_loss)

    def local_test(self, model_global, b_use_test_dataset=False):
        model_global.eval()
        model_global.to(self.device)
        metrics = { 
            'test_correct': 0, 
            'test_loss' : 0, 
            'test_precision': 0,
            'test_recall': 0,
            'test_total' : 0
        }
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = model_global(x)
                loss = self.criterion(pred, target)

                if self.args.dataset == "stackoverflow_lr":
                    predicted = (pred > .5).int()
                    correct = predicted.eq(target).sum(axis = -1).eq(target.size(1)).sum()
                    true_positive = ((target * predicted) > .1).int().sum(axis = -1)
                    precision = true_positive / (predicted.sum(axis = -1) + 1e-13)
                    recall = true_positive / (target.sum(axis = -1)  + 1e-13)
                    metrics['test_precision'] += precision.sum().item()
                    metrics['test_recall'] += recall.sum().item()
                else:
                    _, predicted = torch.max(pred, -1)
                    correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

        return metrics
