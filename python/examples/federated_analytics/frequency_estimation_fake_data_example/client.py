from fedml.fa import init, FARunner
from fedml.fa.data import fa_load_data

if __name__ == "__main__":
    args = init()
    dataset = fa_load_data(args)
    fa_runner = FARunner(args, dataset)
    fa_runner.run()

