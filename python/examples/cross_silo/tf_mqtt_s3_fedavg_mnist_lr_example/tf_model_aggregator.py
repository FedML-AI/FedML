import logging

import tensorflow as tf
import wandb

from fedml import mlops
from fedml.core import ServerAggregator


class TfServerAggregator(ServerAggregator):
    def __init__(self, model, args):
        super().__init__(model, args)

        if args.client_optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=args.learning_rate,
                name='SGD'
            )
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=args.learning_rate,
                name='Adam'
            )

        self.model.compile(optimizer=optimizer, loss=['sparse_categorical_crossentropy'], metrics=['accuracy'])

    def get_model_params(self):
        return self.model.get_weights()

    def set_model_params(self, model_parameters):
        self.model.set_weights(model_parameters)

    def _test(self, test_data, device, args):
        test_results = []

        metrics = {
            "test_acc": 0,
            "test_loss": 0,
            "test_precision": 0,
            "test_recall": 0,
            "test_total": 0,
        }

        for batch_idx, (x, target) in enumerate(test_data):
            # start_time = time.time_ns()
            test_results = self.model.test_on_batch(x=x, y=target, reset_metrics=False)
            # logging.info("test consume time: {}".format(time.time_ns() - start_time))

            metrics["test_total"] += target.get_shape().num_elements()

        metrics["test_acc"] = test_results[1]
        metrics["test_loss"] = test_results[0]

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
        for client_idx in range(len(test_data_local_dict)):
            # test data
            metrics = self._test(test_data_local_dict[client_idx], device, args)
            train_acc, train_num_sample, train_loss = (
                metrics["test_acc"],
                metrics["test_total"],
                metrics["test_loss"],
            )

            logging.info("client_idx = {}, metrics = {}".format(client_idx, metrics))

        # test on training dataset
        if self.args.enable_wandb:
            wandb.log({"Test/Acc": train_acc, "round": args.round_idx})
            wandb.log({"Test/Loss": train_loss, "round": args.round_idx})

        mlops.log({"Test/Acc": train_acc, "round": args.round_idx})
        mlops.log({"Test/Loss": train_loss, "round": args.round_idx})

        stats = {"testing_acc": train_acc, "test_loss": train_loss}
        logging.info(stats)

        return True
