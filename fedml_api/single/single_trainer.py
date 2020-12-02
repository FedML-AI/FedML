import copy
import logging

import numpy as np
import wandb

import torch
from torch import nn




class Single_Trainer(object):
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

        self.model.train()

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        if self.args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)


    def train(self):
        self.model.to(self.device)
        # change to train mode
        self.model.train()

        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(self.train_global):
                # logging.info(images.shape)
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                log_probs = self.model(x)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('Local Training Epoch: {} \tLoss: {:.6f}'.format(
                                 epoch, sum(epoch_loss) / len(epoch_loss)))
            self.local_test_on_all_clients(self.model, round_idx=epoch)

        # return weights, self.local_sample_number

    def local_test_on_all_clients(self, model_global, round_idx):
        logging.info("################local_test_on_test_global_data : {}".format(round_idx))
        train_metrics = {
            'num_samples' : [],
            'num_correct' : [],
            'precisions' : [],
            'recalls' : [],
            'losses' : []
        }

        test_metrics = {
            'num_samples' : [],
            'num_correct' : [],
            'precisions' : [],
            'recalls' : [],
            'losses' : []
        }

        # train data
        train_local_metrics = self.local_test(model_global, False)
        train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
        train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
        train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

        # test data
        test_local_metrics = self.local_test(model_global, True)
        test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
        test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
        test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

        if self.args.dataset == "stackoverflow_lr":
            train_metrics['precisions'].append(copy.deepcopy(train_local_metrics['test_precision']))
            train_metrics['recalls'].append(copy.deepcopy(train_local_metrics['test_recall']))
            test_metrics['precisions'].append(copy.deepcopy(test_local_metrics['test_precision']))
            test_metrics['recalls'].append(copy.deepcopy(test_local_metrics['test_recall']))

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])
        train_precision = sum(train_metrics['precisions']) / sum(train_metrics['num_samples'])
        train_recall = sum(train_metrics['recalls']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])
        test_precision = sum(test_metrics['precisions']) / sum(test_metrics['num_samples'])
        test_recall = sum(test_metrics['recalls']) / sum(test_metrics['num_samples'])

        if self.args.dataset == "stackoverflow_lr":
            stats = {'training_acc': train_acc, 'training_precision': train_precision, 'training_recall': train_recall, 'training_loss': train_loss}
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Pre": train_precision, "round": round_idx})
            wandb.log({"Train/Rec": train_recall, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            logging.info(stats)

            stats = {'test_acc': test_acc, 'test_precision': test_precision, 'test_recall': test_recall, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_precision, "round": round_idx})
            wandb.log({"Test/Rec": test_recall, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            logging.info(stats)

        else:
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            logging.info(stats)

            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            logging.info(stats)



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
            test_data = self.test_global
        else:
            test_data = self.train_global
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
