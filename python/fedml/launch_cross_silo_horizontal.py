import fedml
from .runner import FedMLRunner

from .constants import FEDML_TRAINING_PLATFORM_CROSS_SILO


def run_cross_silo_server():
    """
    Run the server for the FedML Octopus platform using Cross-Silo training.

    This function initializes the server, loads data, and starts training for the Cross-Silo Octopus platform.
    """
    fedml._global_training_type = FEDML_TRAINING_PLATFORM_CROSS_SILO

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


def run_cross_silo_client():
    """
    Run a client for the FedML Octopus platform using Cross-Silo training.

    This function initializes a client, loads data, and starts training for the Cross-Silo Octopus platform.
    """
    global _global_training_type
    _global_training_type = FEDML_TRAINING_PLATFORM_CROSS_SILO

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
