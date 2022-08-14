
import tensorflow as tf
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

    def test(self, test_data, device, args):
        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        for batch_idx, (x, target) in enumerate(test_data):
            x = x.numpy()
            target = target.numpy()
            y_pred = self.model.test_on_batch(x=x, y=target)
            y = self.model.predict(x, verbose=0)
            loss = y_pred[0]
            accuracy = y_pred[1]
            correct = tf.equal(tf.argmax(y, 1), tf.cast(target, tf.int64))

            # metrics["test_correct"] += tf.reduce_mean(tf.cast(correct, tf.float32))
            metrics["test_loss"] += loss
            metrics["test_total"] += 1

        return metrics
