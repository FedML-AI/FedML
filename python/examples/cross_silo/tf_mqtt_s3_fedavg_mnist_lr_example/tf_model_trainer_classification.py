
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
            loss = 0.0
            for batch_idx, (x, labels) in enumerate(train_data):
                y_pred = self.model.train_on_batch(x=x, y=labels, reset_metrics=False)
                loss = y_pred[0]
                accuracy = y_pred[1]

            epoch_loss.append(loss)
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}\tAccuracy: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss), accuracy
                )
            )