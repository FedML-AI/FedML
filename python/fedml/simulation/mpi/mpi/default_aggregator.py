import copy
import logging
import random

import torch
import wandb
from torch import nn

# from ... import mlops
from fedml import mlops
# from ...core.alg_frame.server_aggregator import ServerAggregator
from fedml.ml.aggregator.base_aggregator import BaseServerAggregator


class DefaultServerAggregator(BaseServerAggregator):
    def __init__(self,
        train_global,
        test_global,
        all_train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        client_num,
        device,
        args,
        model,
    ):
        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        super().__init__(model, args, device, client_num)
        self.cpu_transfer = False if not hasattr(self.args, "cpu_transfer") else self.args.cpu_transfer

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def _test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_precision": 0,
            "test_recall": 0,
            "test_total": 0,
        }

        """
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        """
        if args.dataset == "stackoverflow_lr":
            criterion = nn.BCELoss(reduction="sum").to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)  # pylint: disable=E1102

                if args.dataset == "stackoverflow_lr":
                    predicted = (pred > 0.5).int()
                    correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                    true_positive = ((target * predicted) > 0.1).int().sum(axis=-1)
                    precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                    recall = true_positive / (target.sum(axis=-1) + 1e-13)
                    metrics["test_precision"] += precision.sum().item()
                    metrics["test_recall"] += recall.sum().item()
                else:
                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                if len(target.size()) == 1:  #
                    metrics["test_total"] += target.size(0)
                elif len(target.size()) == 2:  # for tasks of next word prediction
                    metrics["test_total"] += target.size(0) * target.size(1)

        model.to("cpu")
        return metrics



    def test_on_server_for_all_clients(self, round_idx):
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))
            # self.aggregator.test_all(
            #     self.train_data_local_dict, self.test_data_local_dict, self.device, self.args,
            # )
            if round_idx == self.args.comm_round - 1:
                self.test(self.test_global, self.device, self.args)
            else:
                self.test(self.val_global, self.device, self.args)
            return
        else:
            mlops.log({"round_idx": round_idx})


    def test(self, test_data, device, args):
        # test data
        test_num_samples = []
        test_tot_corrects = []
        test_losses = []

        metrics = self._test(test_data, device, args)

        test_tot_correct, test_num_sample, test_loss = (
            metrics["test_correct"],
            metrics["test_total"],
            metrics["test_loss"],
        )
        test_tot_corrects.append(copy.deepcopy(test_tot_correct))
        test_num_samples.append(copy.deepcopy(test_num_sample))
        test_losses.append(copy.deepcopy(test_loss))

        # test on test dataset
        test_acc = sum(test_tot_corrects) / sum(test_num_samples)
        test_loss = sum(test_losses) / sum(test_num_samples)
        if self.args.enable_wandb:
            wandb.log({"Test/Acc": test_acc, "round": args.round_idx})
            wandb.log({"Test/Loss": test_loss, "round": args.round_idx})

        mlops.log({"Test/Acc": test_acc, "round": args.round_idx})
        mlops.log({"Test/Loss": test_loss, "round": args.round_idx})

        stats = {"test_acc": test_acc, "test_loss": test_loss}
        logging.info(stats)

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        train_num_samples = []
        train_tot_corrects = []
        train_losses = []
        for client_idx in range(self.args.client_num_in_total):
            # train data
            metrics = self._test(train_data_local_dict[client_idx], device, args)
            train_tot_correct, train_num_sample, train_loss = (
                metrics["test_correct"],
                metrics["test_total"],
                metrics["test_loss"],
            )
            train_tot_corrects.append(copy.deepcopy(train_tot_correct))
            train_num_samples.append(copy.deepcopy(train_num_sample))
            train_losses.append(copy.deepcopy(train_loss))
            # logging.info("testing client_idx = {}".format(client_idx))

        # test on training dataset
        train_acc = sum(train_tot_corrects) / sum(train_num_samples)
        train_loss = sum(train_losses) / sum(train_num_samples)
        if self.args.enable_wandb:
            wandb.log({"Train/Acc": train_acc, "round": args.round_idx})
            wandb.log({"Train/Loss": train_loss, "round": args.round_idx})

        mlops.log({"Train/Acc": train_acc, "round": args.round_idx})
        mlops.log({"Train/Loss": train_loss, "round": args.round_idx})

        stats = {"training_acc": train_acc, "training_loss": train_loss}
        logging.info(stats)

        return True
