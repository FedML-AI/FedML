from fedml import MLEngineBackend
from fedml.fa import init, FARunner
from fedml.fa.data import fa_load_data

if __name__ == "__main__":

    # init FedML framework
    args = init()

    # load data
    dataset = fa_load_data(args)

    # init device
    device = MLEngineBackend.ml_device_type_cpu

    # start training
    fa_runner = FARunner(args, device, dataset)
    fa_runner.run()

