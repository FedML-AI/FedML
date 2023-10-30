import logging
import time

import jax
import numpy as np
import optax
import haiku as hk
import jax.numpy as jnp
from typing import NamedTuple
import wandb

from fedml import mlops
from fedml.core import ServerAggregator


class TrainingState(NamedTuple):
    params: hk.Params
    avg_params: hk.Params
    opt_state: optax.OptState


class JaxHaikuServerAggregator(ServerAggregator):
    static_model = None

    def __init__(self, model, args):
        super().__init__(model, args)
        JaxHaikuServerAggregator.static_model = model
        self.optimizer = None
        self.aggregator_state = self.init_aggregator()

    def get_model_params(self):
        return self.aggregator_state.params

    def set_model_params(self, model_parameters):
        if self.optimizer is None:
            return

        current_opt_state = self.optimizer.init(model_parameters)
        self.aggregator_state = TrainingState(model_parameters, model_parameters, current_opt_state)

    @staticmethod
    def loss(params: hk.Params, x, labels) -> jnp.ndarray:
        """Cross-entropy classification loss with regularization by L2 weight decay."""
        batch_size, *_ = x.shape
        logits = JaxHaikuServerAggregator.static_model.model_network.apply(params, x)
        labels = jax.nn.one_hot(labels, JaxHaikuServerAggregator.static_model.output_dim)

        l2_regularization = 0.5 * sum(
            jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        log_likelihood = jnp.sum(labels * jax.nn.log_softmax(logits))

        return -log_likelihood / batch_size + 1e-4 * l2_regularization

    @staticmethod
    @jax.jit
    def evaluate(params: hk.Params, x, labels) -> jnp.ndarray:
        """Evaluation metrics (classification accuracy)."""
        logits = JaxHaikuServerAggregator.static_model.model_network.apply(params, x)
        predictions = jnp.argmax(logits, axis=-1)
        return jnp.mean(predictions == labels)

    def init_aggregator(self):
        if self.args.client_optimizer == "sgd":
            self.optimizer = optax.sgd(learning_rate=self.args.learning_rate)
        else:
            self.optimizer = optax.adam(learning_rate=self.args.learning_rate)

        initial_opt_state = self.optimizer.init(self.model.initial_params)
        return TrainingState(self.model.initial_params, self.model.initial_params, initial_opt_state)

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
            x = jax.device_put(x, device)
            target = jax.device_put(target, device)

            accuracy = np.array(
                JaxHaikuServerAggregator.evaluate(self.aggregator_state.params, x, target)).item()
            loss = JaxHaikuServerAggregator.loss(self.aggregator_state.params, x, target).item()

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
        for client_idx in range(len(test_data_local_dict)):
            # test data
            metrics = self._test(test_data_local_dict[client_idx], device, args)
            train_acc, train_num_sample, train_loss = (
                metrics["test_acc"],
                metrics["test_total"],
                metrics["test_loss"],
            )

            # logging.info("client_idx = {}, metrics = {}".format(client_idx, metrics))

        # test on training dataset
        if self.args.enable_wandb:
            wandb.log({"Test/Acc": train_acc, "round": args.round_idx})
            wandb.log({"Test/Loss": train_loss, "round": args.round_idx})

        mlops.log({"Test/Acc": train_acc, "round": args.round_idx})
        mlops.log({"Test/Loss": train_loss, "round": args.round_idx})

        stats = {"testing_acc": train_acc, "testing_loss": train_loss}
        logging.info(stats)

        return True
