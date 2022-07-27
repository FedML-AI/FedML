import logging

import fedml
from fedml import FedMLRunner
from fedml.core import FedMLExecutor, Params, FedMLAlgorithmFlow


class Client(FedMLExecutor):
    def __init__(self, args):
        self.args = args
        id = args.rank
        neighbor_id_list = [0]
        super().__init__(id, neighbor_id_list)

    def local_training(self):
        logging.info("local_training")

        params = Params()
        params.add("whatever_key_as_you_like_1", "any_value")
        params.add("whatever_key_as_you_like_2", 2)
        return params


class Server(FedMLExecutor):
    def __init__(self, args):
        self.args = args
        id = args.rank
        neighbor_id_list = [1, 2]
        super().__init__(id, neighbor_id_list)

        self.round_idx = 0

    def init_global_model(self):
        logging.info("init_global_model")
        params = Params()
        return params

    def server_aggregate(self):
        logging.info("server_aggregate")
        params = self.get_params()
        value1 = params.get("whatever_key_as_you_like_1")
        logging.info("value1 = {}".format(value1))
        self.round_idx += 1
        if self.round_idx == self.loop_times:
            params = Params()
            return params

    def final_eval(self):
        logging.info("final_eval")


if __name__ == "__main__":
    args = fedml.init()

    # init device
    # device = fedml.device.get_device(args)
    device = None

    # load data
    # dataset, output_dim = fedml.data.load(args)
    dataset, output_dim = None, 10

    # load model
    # model = fedml.model.create(args, output_dim)
    model = None

    if args.rank == 0:
        executor = Server(args)
    else:
        executor = Client(args)

    fedml_alg_flow = FedMLAlgorithmFlow(args, executor, loop_times=2)
    fedml_alg_flow.add_flow("init_global_model", Server.init_global_model)
    fedml_alg_flow.add_flow("local_training", Client.local_training, flow_tag=FedMLAlgorithmFlow.LOOP_START)
    fedml_alg_flow.add_flow("server_aggregate", Server.server_aggregate, flow_tag=FedMLAlgorithmFlow.LOOP_END)
    fedml_alg_flow.add_flow("final_eval", Server.final_eval)
    fedml_alg_flow.build()

    fedml_runner = FedMLRunner(args, device, dataset, model, algorithm_flow=fedml_alg_flow)
    fedml_runner.run()
