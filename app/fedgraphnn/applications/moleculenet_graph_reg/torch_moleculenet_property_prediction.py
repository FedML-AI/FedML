import fedml
from fedml.simulation import SimulatorMPI

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedgraphnn.data.load(args)

    # load model
    model, trainer = fedgraphnn.model.create(args, output_dim)

    # start training
    simulator = SimulatorMPI(args, device, dataset, model, trainer)
    simulator.run()