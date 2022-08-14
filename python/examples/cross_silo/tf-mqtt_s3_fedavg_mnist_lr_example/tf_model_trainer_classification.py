
import logging

from fedml.core import ClientTrainer
import tensorflow as tf


class TfModelTrainerCLS(ClientTrainer):
    def get_model_params(self):
        return self.model.get_weights()

    def set_model_params(self, model_parameters):
        self.model.set_weights(model_parameters)

    def train(self, train_data, device, args):
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

        epoch_loss = []
        accuracy = 0.0
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x = x.numpy()
                labels = labels.numpy()
                y_pred = self.model.train_on_batch(x=x, y=labels)
                loss = y_pred[0]
                accuracy = y_pred[1]
                batch_loss.append(loss)

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}\tAccuracy: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss), accuracy
                )
            )

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
            metrics["test_loss"] += loss * target.size(0)
            metrics["test_total"] += target.size(0)

        return metrics
