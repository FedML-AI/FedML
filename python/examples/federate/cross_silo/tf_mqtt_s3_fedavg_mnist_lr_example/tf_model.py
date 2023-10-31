import tensorflow as tf


class LogisticRegressionModel(tf.keras.Model):
    def __init__(self, input_dim, out_dim, name=None):
        super(LogisticRegressionModel, self).__init__(name=name)
        self.output_dim = out_dim
        self.layer1 = tf.keras.layers.Dense(out_dim, input_shape=(input_dim,), activation="sigmoid")
        self.layer1.build(input_shape=(input_dim,))

    def call(self, x):
        return self.layer1(x)

    def get_config(self):
        return {"output_dim": self.output_dim, "name": self.name}


def create_model(input_dim, out_dim):
    # tf.compat.v1.disable_eager_execution()
    client_model = LogisticRegressionModel(input_dim, out_dim)
    return client_model
