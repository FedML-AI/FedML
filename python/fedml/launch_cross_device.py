import fedml

from .constants import FEDML_TRAINING_PLATFORM_CROSS_DEVICE


def run_mnn_server():
    from .cross_device import ServerMNN

    """FedML BeeHive"""
    fedml._global_training_type = FEDML_TRAINING_PLATFORM_CROSS_DEVICE

    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    server = ServerMNN(args, device, dataset, model)
    server.run()
