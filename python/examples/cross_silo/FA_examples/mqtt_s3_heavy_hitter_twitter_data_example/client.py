from fedml.fa import init, FARunner
from fedml.fa.data import fa_load_data

if __name__ == "__main__":

    # init FedML framework
    args = init()

    # load data
    dataset = fa_load_data(args)

    # start training
    fa_runner = FARunner(args, dataset)
    fa_runner.run()

