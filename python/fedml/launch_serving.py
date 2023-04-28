import fedml
from .runner import FedMLRunner

from .constants import FEDML_TRAINING_PLATFORM_SERVING


def run_model_serving_server():
    """FedML Model Serving"""
    fedml._global_training_type = FEDML_TRAINING_PLATFORM_SERVING

    args = fedml.init()
    args.role = "server"

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()


def run_model_serving_client():
    """FedML Model Serving"""
    fedml._global_training_type = FEDML_TRAINING_PLATFORM_SERVING

    args = fedml.init()
    args.role = "client"

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
