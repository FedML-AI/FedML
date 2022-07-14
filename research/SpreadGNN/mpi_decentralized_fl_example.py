import fedml
from fedml import SimulatorMPI

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # start training
    simulator = SimulatorMPI(args, None, None, None)
    simulator.run()
