from mxnet.gluon import nn


class LogisticRegressionModel(nn.Block):
    def __init__(self, input_dim, out_dim, name=None, **kwargs):
        # Run `nn.Block`'s init method
        super(LogisticRegressionModel, self).__init__(**kwargs)
        self.output_dim = out_dim
        self.name = name

        self.lay1 = nn.Dense(out_dim, activation='sigmoid')

    def forward(self, x):
        return self.lay1(x)

    def get_config(self):
        return {"output_dim": self.output_dim, "name": self.name}


def create_model(input_dim, out_dim):
    client_model = LogisticRegressionModel(input_dim, out_dim)
    client_model.initialize()
    return client_model
