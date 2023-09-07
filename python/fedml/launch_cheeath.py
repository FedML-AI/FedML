import fedml
from .runner import FedMLRunner

from .constants import FEDML_TRAINING_PLATFORM_CHEETAH


def run_cheetah_server():
    """
    Run the server for the FedML Cheetah platform.

    This function initializes the server, loads data, and starts training using the Cheetah server.
    """
    fedml._global_training_type = FEDML_TRAINING_PLATFORM_CHEETAH

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


def run_cheetah_client():
    """
    Run a client for the FedML Cheetah platform.

    This function initializes a client, loads data, and starts training using the Cheetah client.
    """
    fedml._global_training_type = FEDML_TRAINING_PLATFORM_CHEETAH

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
