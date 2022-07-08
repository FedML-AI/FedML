import fedml

from .constants import FEDML_TRAINING_PLATFORM_CROSS_SILO


def run_hierarchical_cross_silo_server():
    from .cross_silo.hierarchical import Server

    """FedML Octopus"""
    fedml._global_training_type = FEDML_TRAINING_PLATFORM_CROSS_SILO

    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    server = Server(args, device, dataset, model)
    server.run()


def run_hierarchical_cross_silo_client():
    from .cross_silo.hierarchical import Client

    """FedML Octopus"""
    fedml._global_training_type = FEDML_TRAINING_PLATFORM_CROSS_SILO

    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load_cross_silo(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    client = Client(args, device, dataset, model)
    client.run()
