from collections import OrderedDict

from mxnet import np as mx_np
from mxnet.gluon import nn


class LogisticRegressionModel(nn.Block):
    def __init__(self, input_dim, out_dim):
        # Run `nn.Block`'s init method
        super().__init__()
        self.output_dim = out_dim
        self.input_dim = input_dim

        self.layer1 = nn.Dense(out_dim, activation="sigmoid")
        self.layer1.initialize()

    def forward(self, x):
        return self.layer1(x)

    def get_params(self):
        param = OrderedDict()
        param["weight"] = list()
        for model_val in self.layer1.collect_params(".*weight").values():
            model_data = model_val.data()
            param["weight"].append(model_data.asnumpy())

        param["bias"] = list()
        for model_val in self.layer1.collect_params(".*bias").values():
            model_data = model_val.data()
            param["bias"].append(model_data.asnumpy())
        return param

    def get_layer_params(self):
        return self.layer1.collect_params()

    def set_params(self, in_params):
        weight_params = in_params["weight"]
        bias_params = in_params["bias"]
        params = zip(self.layer1.collect_params(".*weight").keys(), weight_params)
        for key, value in params:
            data = mx_np.array(value)
            self.layer1.collect_params()[key].set_data(data)

        params = zip(self.layer1.collect_params(".*bias").keys(), bias_params)
        for key, value in params:
            data = mx_np.array(value)
            self.layer1.collect_params()[key].set_data(data)


def create_model(input_dim, out_dim):
    client_model = LogisticRegressionModel(input_dim, out_dim)
    client_model.initialize()
    return client_model
