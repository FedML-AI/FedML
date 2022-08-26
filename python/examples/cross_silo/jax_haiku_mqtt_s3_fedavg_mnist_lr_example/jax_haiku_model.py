
import haiku as hk
import jax
import jax.numpy as jnp


class LogisticRegressionModelImpl(hk.Module):
    def __init__(self, input_dim, out_dim, name=None):
        super().__init__(name=name)
        self.output_dim = out_dim

    def __call__(self, x):
        # j, k = x.shape[-1], self.output_dim
        # w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(j))
        # w = hk.get_parameter("w", shape=[j, k], dtype=x.dtype, init=w_init)
        # b = hk.get_parameter("b", shape=[k], dtype=x.dtype, init=jnp.ones)
        # return jax.nn.sigmoid(jnp.dot(x, w) + b)
        x = hk.Flatten()(x)
        x = hk.Linear(self.output_dim)(x)
        x = jax.nn.sigmoid(x)
        return x

    def get_config(self):
        return {"output_dim": self.output_dim, "name": self.name}


class LogisticRegressionModel:
    def __init__(self, input_dim, out_dim, name=None):
        self.output_dim = out_dim
        self.name = name

        def model_network_fn(x):
            client_model = LogisticRegressionModelImpl(input_dim, out_dim)
            return client_model(x)

        self.model_network = hk.without_apply_rng(hk.transform(model_network_fn))
        init_x = jnp.ones(input_dim).block_until_ready()
        self.initial_params = self.model_network.init(jax.random.PRNGKey(seed=0), init_x)


def create_model(input_dim, out_dim):
    client_model = LogisticRegressionModel(input_dim, out_dim)
    return client_model

