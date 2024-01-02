import fedml
from fedml import FedMLRunner

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # start training
    fedml_runner = FedMLRunner(args, None, None, None)
    fedml_runner.run()
