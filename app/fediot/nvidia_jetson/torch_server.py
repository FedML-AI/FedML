import fedml
from fedml.cross_silo import Server
from data.data_loader import load_data
from model.autoencoder import AutoEncoder


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args)

    # load model
    model = AutoEncoder(output_dim)

    # start training
    server = Server(args, device, dataset, model)
    server.run()
