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

        self.device = None
        self.dataset = None
        self.model = None

    def init(self, device, dataset, model):
        self.device = device
        self.dataset = dataset
        self.model = model

    def local_training(self):
        logging.info("local_training start")
        params = self.get_params()
        model_params = params.get(Params.KEY_MODEL_PARAMS)
        return params

    def handle_init_global_model(self):
        received_params = self.get_params()
        model_params = received_params.get(Params.KEY_MODEL_PARAMS)

        params = Params()
        params.add(Params.KEY_MODEL_PARAMS, model_params)
        return params


class Server(FedMLExecutor):
    def __init__(self, args):
        self.args = args
        id = args.rank
        neighbor_id_list = [1, 2]
        super().__init__(id, neighbor_id_list)

        self.device = None
        self.dataset = None
        self.model = None

        self.round_idx = 0

        self.client_count = 0
        self.client_num = 2

    def init(self, device, dataset, model):
        self.device = device
        self.dataset = dataset
        self.model = model

    def init_global_model(self):
        logging.info("init_global_model")
        params = Params()
        params.add(Params.KEY_MODEL_PARAMS, self.model.state_dict())
        return params

    def server_aggregate(self):
        logging.info("server_aggregate")
        params = self.get_params()
        model_params = params.get(Params.KEY_MODEL_PARAMS)
        # logging.info("value1 = {}".format(value1))
        self.round_idx += 1
        self.client_count += 1
        if self.client_count == self.client_num:
            self.client_count = 0
            params = Params()
            params.add(Params.KEY_MODEL_PARAMS, model_params)
            return params

    def final_eval(self):
        logging.info("final_eval")


if __name__ == "__main__":
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    if args.rank == 0:
        executor = Server(args)
        executor.init(device, dataset, model)
    else:
        executor = Client(args)
        executor.init(device, dataset, model)

    fedml_alg_flow = FedMLAlgorithmFlow(args, executor)
    fedml_alg_flow.add_flow("init_global_model", Server.init_global_model)
    fedml_alg_flow.add_flow("handle_init", Client.handle_init_global_model)
    for round_idx in range(args.comm_round):
        fedml_alg_flow.add_flow("local_training", Client.local_training)
        fedml_alg_flow.add_flow("server_aggregate", Server.server_aggregate)
    fedml_alg_flow.add_flow("final_eval", Server.final_eval)
    fedml_alg_flow.build()

    fedml_runner = FedMLRunner(args, device, dataset, model, algorithm_flow=fedml_alg_flow)
    fedml_runner.run()
