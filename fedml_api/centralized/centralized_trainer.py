import copy
import logging

import torch
import wandb
from torch import nn
from torch.nn.parallel import DistributedDataParallel

class CentralizedTrainer(object):
    r"""
    This class is used to train federated non-IID dataset in a centralized way
    """

    def __init__(self, dataset, model, device, args):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model = model
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        if self.args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)

    def train(self):
        for epoch in range(self.args.epochs):
            if self.args.data_parallel == 1:
                self.train_global.sampler.set_epoch(epoch)
            self.train_impl(epoch)
            self.eval_impl(epoch)

    def train_impl(self, epoch_idx):
        self.model.train()
        for batch_idx, (x, labels) in enumerate(self.train_global):
            # logging.info(images.shape)
            x, labels = x.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            log_probs = self.model(x)
            loss = self.criterion(log_probs, labels)
            loss.backward()
            self.optimizer.step()
            logging.info('Local Training Epoch: {} {}-th iters\t Loss: {:.6f}'.format(epoch_idx,
                                                                                      batch_idx, loss.item()))

    def eval_impl(self, epoch_idx):
        # train
        if epoch_idx % self.args.frequency_of_train_acc_report == 0:
            self.test_on_all_clients(b_is_train=True, epoch_idx=epoch_idx)

        # test
        if epoch_idx % self.args.frequency_of_train_acc_report == 0:
            self.test_on_all_clients(b_is_train=False, epoch_idx=epoch_idx)

    def test_on_all_clients(self, b_is_train, epoch_idx):
        self.model.eval()
        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_precision': 0,
            'test_recall': 0,
            'test_total': 0
        }
        if b_is_train:
            test_data = self.train_global
        else:
            test_data = self.test_global
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, target)

                if self.args.dataset == "stackoverflow_lr":
                    predicted = (pred > .5).int()
                    correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                    true_positive = ((target * predicted) > .1).int().sum(axis=-1)
                    precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                    recall = true_positive / (target.sum(axis=-1) + 1e-13)
                    metrics['test_precision'] += precision.sum().item()
                    metrics['test_recall'] += recall.sum().item()
                else:
                    _, predicted = torch.max(pred, -1)
                    correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        if self.args.rank == 0:
            self.save_log(b_is_train=b_is_train, metrics=metrics, epoch_idx=epoch_idx)

    def save_log(self, b_is_train, metrics, epoch_idx):
        prefix = 'Train' if b_is_train else 'Test'

        all_metrics = {
            'num_samples': [],
            'num_correct': [],
            'precisions': [],
            'recalls': [],
            'losses': []
        }

        all_metrics['num_samples'].append(copy.deepcopy(metrics['test_total']))
        all_metrics['num_correct'].append(copy.deepcopy(metrics['test_correct']))
        all_metrics['losses'].append(copy.deepcopy(metrics['test_loss']))

        if self.args.dataset == "stackoverflow_lr":
            all_metrics['precisions'].append(copy.deepcopy(metrics['test_precision']))
            all_metrics['recalls'].append(copy.deepcopy(metrics['test_recall']))

        # performance on all clients
        acc = sum(all_metrics['num_correct']) / sum(all_metrics['num_samples'])
        loss = sum(all_metrics['losses']) / sum(all_metrics['num_samples'])
        precision = sum(all_metrics['precisions']) / sum(all_metrics['num_samples'])
        recall = sum(all_metrics['recalls']) / sum(all_metrics['num_samples'])

        if self.args.dataset == "stackoverflow_lr":
            stats = {prefix + '_acc': acc, prefix + '_precision': precision, prefix + '_recall': recall,
                     prefix + '_loss': loss}
            wandb.log({prefix + "/Acc": acc, "epoch": epoch_idx})
            wandb.log({prefix + "/Pre": precision, "epoch": epoch_idx})
            wandb.log({prefix + "/Rec": recall, "epoch": epoch_idx})
            wandb.log({prefix + "/Loss": loss, "epoch": epoch_idx})
            logging.info(stats)
        else:
            stats = {prefix + '_acc': acc, prefix + '_loss': loss}
            wandb.log({prefix + "/Acc": acc, "epoch": epoch_idx})
            wandb.log({prefix + "/Loss": loss, "epoch": epoch_idx})
            logging.info(stats)

        stats = {prefix + '_acc': acc, prefix + '_loss': loss}
        wandb.log({prefix + "/Acc": acc, "epoch": epoch_idx})
        wandb.log({prefix + "/Loss": loss, "epoch": epoch_idx})
        logging.info(stats)