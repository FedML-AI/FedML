
import fedml
from fedml import FedMLRunner
from fedml.data.MNIST.data_loader import download_mnist, load_partition_data_mnist
from jax_haiku_model_trainer_classification import JaxHaikuModelTrainerCLS
import haiku as hk
import jax
import jax.numpy as jnp


def load_data(args):
    download_mnist(args.data_cache_dir)
    fedml.logging.info("load_data. dataset_name = %s" % args.dataset)

    """
    Please read through the data loader at to see how to customize the dataset for FedML framework.
    """
    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_partition_data_mnist(
        args,
        args.batch_size,
        train_path=args.data_cache_dir + "/MNIST/train",
        test_path=args.data_cache_dir + "/MNIST/test",
    )
    """
    For shallow NN or linear models, 
    we uniformly sample a fraction of clients each round (as the original FedAvg paper)
    """
    args.client_num_in_total = client_num
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num


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


def create_model_trainer(in_model, in_args):
    model_trainer = JaxHaikuModelTrainerCLS(in_model, in_args)
    return model_trainer


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()
    setattr(args, "run_id", "1979")

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args)

    # load model (the size of MNIST image is 28 x 28)
    model = create_model(28 * 28, output_dim)

    # create model trainer
    trainer = create_model_trainer(model, args)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, client_trainer=trainer)
    fedml_runner.run()
