import logging
import numpy as np

import fedml
from fedml import FedMLRunner
from fedml.core import FedMLExecutor, Params, FedMLAlgorithmFlow

from fedml import mlops
from fedml.ml.aggregator.agg_operator import FedMLAggOperator
from fedml.ml.trainer.trainer_creator import create_model_trainer
from fedml.ml.aggregator.aggregator_creator import create_server_aggregator


from .message_define import MyMessage


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
        self.model_trainer = create_model_trainer(model, self.args)
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict


    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]


    def local_training(self):
        logging.info("local_training start")
        params = self.get_params()
        model_params = params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.trainer.update_dataset(int(client_index))
        self.model_trainer.set_id(client_index)

        self.model_trainer.set_model_params(model_params)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()

        params = Params()
        params.add(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        params.add(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, self.local_sample_num)
        return params

    def handle_init_global_model(self):
        received_params = self.get_params()
        model_params = received_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        # handled by local_training
        params = Params()
        params.add(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, model_params)
        params.add(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, client_index)
        return params

    def client_test(self):
        logging.info("local_training start")
        params = self.get_params()
        model_params = params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.model_trainer.set_model_params(model_params)
        metrics = self.model_trainer.test(test_data, device, args)
        params = Params()
        params.add(MyMessage.MSG_ARG_KEY_TEST_METRIC, metrics)
        return params


class Server(FedMLExecutor):
    def __init__(self, args):
        self.args = args
        id = args.rank
        # neighbor_id_list = [1, 2]
        neighbor_id_list = list(range(args.client_num_per_round))
        super().__init__(id, neighbor_id_list)

        self.device = None
        self.dataset = None
        self.model = None

        self.round_idx = 0

        self.client_count = 0
        self.worker_num = self.args.client_num_per_round
        # self.client_num = self.args.client_num_per_round
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False


    def init(self, device, dataset, model):
        self.device = device
        self.dataset = dataset
        self.model = model
        self.model_trainer = create_model_trainer(model, self.args)
        # self.aggregator = create_server_aggregator(model, args)


    def check_whether_all_receive(self):
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True


    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                range(client_num_in_total), num_clients, replace=False
            )
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes


    def init_global_model(self):
        logging.info("init_global_model")
        params = Params()
        params.add(Params.KEY_MODEL_PARAMS, self.model.state_dict())
        return params

    def server_aggregate(self):
        logging.info("server_aggregate")

        params = self.get_params()
        sender_id = params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        logging.info("add_model. index = %d" % sender_id - 1)
        self.model_dict[sender_id - 1] = model_params
        self.sample_num_dict[sender_id - 1] = local_sample_number
        self.flag_client_model_uploaded_dict[sender_id - 1] = True
        b_all_received = self.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))

        if b_all_received:

            global_model_params = FedMLAggOperator.agg(self.args, w_locals)
            w_locals = []
            for idx in range(self.worker_num):
                w_locals.append((self.sample_num_dict[idx], self.model_dict[idx]))
            self.model_trainer.set_model_params(global_model_params)
            # self.aggregator.test_on_server_for_all_clients(self.args.round_idx)

        # use start_round to return params for communication.


    def start_round(self):

        global_model_params = self.model_trainer.get_model_params()
        client_indexes = self.aggregator.client_sampling(
            self.args.round_idx,
            self.args.client_num_in_total,
            self.args.client_num_per_round,
        )
        params = Params()
        params.add(Params.KEY_MODEL_PARAMS, global_model_params)
        params.add(Params.MSG_ARG_KEY_CLIENT_INDEX, 
        return params




    def server_test(self):
        metrics = self.model_trainer.test(test_data, device, args)

    def start_distributed_test(self):
        pass


    def distributed_test(self):
        params = self.get_params()
        metrics = params.get(MyMessage.MSG_ARG_KEY_TEST_METRIC)




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
    fedml_alg_flow.add_flow("start_round", Server.start_round)
    fedml_alg_flow.add_flow("handle_init", Client.handle_init_global_model)
    for round_idx in range(args.comm_round):
        fedml_alg_flow.add_flow("local_training", Client.local_training)
        fedml_alg_flow.add_flow("server_aggregate", Server.server_aggregate)
        if (
            round_idx % args.frequency_of_the_test == 0
            or round_idx == args.comm_round - 1
        ):
            fedml_alg_flow.add_flow("start_distributed_test", Server.start_distributed_test)
            fedml_alg_flow.add_flow("client_test", Client.client_test)
            fedml_alg_flow.add_flow("distributed_test", Server.distributed_test)
        fedml_alg_flow.add_flow("start_round", Server.start_round)


    fedml_alg_flow.add_flow("final_eval", Server.final_eval)
    fedml_alg_flow.build()

    fedml_runner = FedMLRunner(args, device, dataset, model, algorithm_flow=fedml_alg_flow)
    fedml_runner.run()
