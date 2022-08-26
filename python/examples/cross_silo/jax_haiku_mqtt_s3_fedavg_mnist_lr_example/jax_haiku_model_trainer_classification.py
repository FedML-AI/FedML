import logging

import numpy as np
from fedml.core import ClientTrainer
from typing import NamedTuple
import optax
import haiku as hk
import jax.numpy as jnp
import jax


class TrainingState(NamedTuple):
    params: hk.Params
    avg_params: hk.Params
    opt_state: optax.OptState


class JaxHaikuModelTrainerCLS(ClientTrainer):
    static_model = None
    static_optimizer = None

    def __init__(self, model, args):
        super().__init__(model, args)
        JaxHaikuModelTrainerCLS.static_model = model
        self.optimizer = None
        self.training_state = self.init_trainer()
        JaxHaikuModelTrainerCLS.static_optimizer = self.optimizer

    def get_model_params(self):
        return self.training_state.params

    def set_model_params(self, model_parameters):
        if self.optimizer is None:
            return

        current_opt_state = self.optimizer.init(model_parameters)
        self.training_state = TrainingState(model_parameters, model_parameters, current_opt_state)

    def init_trainer(self):
        if self.args.client_optimizer == "sgd":
            self.optimizer = optax.sgd(learning_rate=self.args.learning_rate)
        else:
            self.optimizer = optax.adam(learning_rate=self.args.learning_rate)

        initial_opt_state = self.optimizer.init(self.model.initial_params)
        return TrainingState(self.model.initial_params, self.model.initial_params, initial_opt_state)

    @staticmethod
    def loss(params: hk.Params, x, labels) -> jnp.ndarray:
        """Cross-entropy classification loss with regularization by L2 weight decay."""
        batch_size, *_ = x.shape
        logits = JaxHaikuModelTrainerCLS.static_model.model_network.apply(params, x)
        labels = jax.nn.one_hot(labels, JaxHaikuModelTrainerCLS.static_model.output_dim)

        l2_regularization = 0.5 * sum(
            jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        log_likelihood = jnp.sum(labels * jax.nn.log_softmax(logits))

        return -log_likelihood / batch_size + 1e-4 * l2_regularization

    @staticmethod
    @jax.jit
    def evaluate(params: hk.Params, x, labels) -> jnp.ndarray:
        """Evaluation metrics (classification accuracy)."""
        logits = JaxHaikuModelTrainerCLS.static_model.model_network.apply(params, x)
        predictions = jnp.argmax(logits, axis=-1)
        return jnp.mean(predictions == labels)

    @staticmethod
    @jax.jit
    def update(state: TrainingState, x, labels) -> TrainingState:
        """Learning rule (stochastic gradient descent)."""
        grads = jax.grad(JaxHaikuModelTrainerCLS.loss)(state.params,
                                                       x, labels)
        updates, opt_state = JaxHaikuModelTrainerCLS.static_optimizer.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        # Compute avg_params, the exponential moving average of the "live" params.
        # We use this only for evaluation (cf. https://doi.org/10.1137/0330046).
        avg_params = optax.incremental_update(
            params, state.avg_params, step_size=0.001)
        return TrainingState(params, avg_params, opt_state)

    def train(self, train_data, device, args):
        epoch_acc = []
        epoch_loss = []
        for epoch in range(args.epochs):
            accuracy = 0.0
            loss = 0.0
            for batch_idx, (x, labels) in enumerate(train_data):
                # Do SGD on a batch of training samples.
                x = jax.device_put(x, device)
                labels = jax.device_put(labels, device)

                self.training_state = JaxHaikuModelTrainerCLS.update(self.training_state,
                                                                     x, labels)

                # Evaluate accuracy
                accuracy = np.array(
                    JaxHaikuModelTrainerCLS.evaluate(self.training_state.params, x,
                                                     labels)).item()
                loss = JaxHaikuModelTrainerCLS.loss(self.training_state.params, x, labels)

            epoch_acc.append(accuracy)
            epoch_loss.append(loss)
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}\tAccuracy: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss),  sum(epoch_acc) / len(epoch_acc)
                )
            )
