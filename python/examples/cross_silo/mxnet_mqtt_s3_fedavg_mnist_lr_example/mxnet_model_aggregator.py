import logging
import time

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
import wandb

from fedml import mlops
from fedml.core import ServerAggregator


class MxServerAggregator(ServerAggregator):
    static_model = None

    def __init__(self, model, args):
        super().__init__(model, args)
        self.optimizer = None
        self.mx_trainer = None
        self.mx_metric = None
        self.mx_loss = None
        self.init_aggregator()

    def get_model_params(self):
        return self.model.collect_params()

    def set_model_params(self, model_parameters):
        if self.optimizer is None:
            return

        self.model.share_parameters(model_parameters)

    def init_aggregator(self):
        # Trainer is for updating parameters with gradient.
        self.mx_metric = mx.gluon.metric.Accuracy()
        self.mx_loss = gluon.loss.SoftmaxCrossEntropyLoss()

    def _test(self, test_data, device, args):
        metrics = {
            "test_acc": 0,
            "test_loss": 0,
            "test_precision": 0,
            "test_recall": 0,
            "test_total": 0,
        }

        batch_acc = []
        batch_loss = []
        for batch_idx, (x, target) in enumerate(test_data):
            # start_time = time.time_ns()
            with device:
                x = x.numpy()
                target = target.numpy()

            output = self.model(x)
            self.mx_metric.update([target], [output])
            loss_func = self.mx_loss(output, target)
            loss = loss_func.mean()
            accuracy = self.mx_metric.get()

            # logging.info("test consume time: {}".format(time.time_ns() - start_time))

            metrics["test_total"] += target.size

            batch_acc.append(accuracy)
            batch_loss.append(loss)

        metrics["test_acc"] = sum(batch_acc) / len(batch_acc)
        metrics["test_loss"] = sum(batch_loss) / len(batch_loss)

        return metrics

    def test(self, test_data, device, args):
        # test data
        metrics = self._test(test_data, device, args)

        test_acc, test_num_sample, test_loss = (
            metrics["test_acc"],
            metrics["test_total"],
            metrics["test_loss"],
        )

        # test on test dataset
        if self.args.enable_wandb:
            wandb.log({"Test/Acc": test_acc, "round": args.round_idx})
            wandb.log({"Test/Loss": test_loss, "round": args.round_idx})

        mlops.log({"Test/Acc": test_acc, "round": args.round_idx})
        mlops.log({"Test/Loss": test_loss, "round": args.round_idx})

        stats = {"test_acc": test_acc, "test_loss": test_loss}
        logging.info(stats)

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        train_acc = 0
        train_loss = 0
        for client_idx in range(self.args.client_num_in_total):
            # train data
            metrics = self._test(train_data_local_dict[client_idx], device, args)
            train_acc, train_num_sample, train_loss = (
                metrics["test_acc"],
                metrics["test_total"],
                metrics["test_loss"],
            )

            # logging.info("client_idx = {}, metrics = {}".format(client_idx, metrics))

        # test on training dataset
        if self.args.enable_wandb:
            wandb.log({"Train/Acc": train_acc, "round": args.round_idx})
            wandb.log({"Train/Loss": train_loss, "round": args.round_idx})

        mlops.log({"Train/Acc": train_acc, "round": args.round_idx})
        mlops.log({"Train/Loss": train_loss, "round": args.round_idx})

        stats = {"training_acc": train_acc, "training_loss": train_loss}
        logging.info(stats)

        return True
