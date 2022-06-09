import fedml
from fedml.cross_silo import Client
from data.data_loader import load_data
from model.autoencoder import AutoEncoder


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args)

    # load model (the size of MNIST image is 28 x 28)
    model = AutoEncoder(output_dim)

    # start training
    client = Client(args, device, dataset, model)
    client.run()
