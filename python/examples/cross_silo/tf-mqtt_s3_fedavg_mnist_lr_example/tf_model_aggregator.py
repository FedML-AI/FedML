import logging

import tensorflow as tf

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

    def test(self, test_data, device, args):
        test_results = []
        for batch_idx, (x, target) in enumerate(test_data):
            x = x.numpy()
            target = target.numpy()
            test_results = self.model.test_on_batch(x=x, y=target, reset_metrics=False)
        logging.info("test_results = {}".format(test_results))
        mlops.log({"Test/Loss": test_results[0], "round": args.round_idx})
        mlops.log({"Test/Acc": test_results[1], "round": args.round_idx})
