import logging
import time
from math import cos

from fedml.ml.aggregator.server_optimizer_creator import create_server_optimizer
from fedml.ml.aggregator.hierarchical_local_aggregator import HierarchicalLocalAggregator
from fedml.ml.ml_message import MLMessage


from .message_define import MyMessage
from .utils import transform_list_to_tensor
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message

from fedml.ml.trainer.local_cache import FedMLLocalCache


class ClientManager(FedMLCommManager):
    def __init__(
        self,
        args,
        trainer,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
    ):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.args.round_idx = 0
        self.worker_id = self.rank - 1
        # self.hierarchical_aggregator = HierarchicalLocalAggregator(trainer.model, args, trainer.device)
        self.hierarchical_aggregator = HierarchicalLocalAggregator(args, trainer.device)

        # if hasattr(self.args, "aggregate_seq") and self.args.aggregate_seq:

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.handle_message_receive_model_from_server,
        )

    def handle_message_init(self, msg_params):
        # global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_cache_path = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_CACHE_PATH)
        FedMLLocalCache.init(self.args, root=self.args.local_cache_root, path=local_cache_path)

        server_result = msg_params.get(MyMessage.MSG_ARG_KEY_SERVER_RESULT)
        # client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        client_schedule = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_SCHEDULE)
        client_indexes = client_schedule[self.worker_id]

        self.args.round_idx = 0
        self.__train(server_result, client_indexes)

    def start_training(self):
        self.args.round_idx = 0
        # self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        # logging.info("handle_message_receive_model_from_server.")
        # model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        server_result = msg_params.get(MyMessage.MSG_ARG_KEY_SERVER_RESULT)
        # client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        client_schedule = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_SCHEDULE)
        client_indexes = client_schedule[self.worker_id]

        self.__train(server_result, client_indexes)
        if self.args.round_idx == self.num_rounds - 1:
            # post_complete_message_to_sweep_process(self.args)
            # self.finish()
            pass


    def send_local_agg_result_to_server(self, receive_id, local_agg_client_result,
                local_sample_num_dict, client_runtime_info):
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_RESULT, local_agg_client_result)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_RUNTIME_INFO, client_runtime_info)
        self.send_message(message)

    def __train(self, server_result, client_indexes):
        logging.info(f"#######training########### worker:{self.worker_id} round_id = {self.args.round_idx}")

        if hasattr(self.args, "simulation_gpu_hetero") and self.args.simulation_gpu_hetero:
            # runtime_speed_ratio
            # runtime_speed_ratio * t_train - t_train
            # time.sleep(runtime_speed_ratio * t_train - t_train)
            simulation_gpu_hetero = self.args.simulation_gpu_hetero
            runtime_speed_ratio_gpu = self.args.gpu_hetero_ratio * self.worker_id / self.args.worker_num

        if hasattr(self.args, "simulation_environment_hetero") and self.args.simulation_environment_hetero:
            # runtime_speed_ratio
            # runtime_speed_ratio * t_train - t_train
            # time.sleep(runtime_speed_ratio * t_train - t_train)
            if self.args.simulation_environment_hetero == "cos":
                runtime_speed_ratio_env = self.args.environment_hetero_ratio * \
                    (1 + cos(self.args.round_idx / self.num_rounds*3.1415926 + self.worker_id))
            else:
                raise NotImplementedError

        training_num_in_round = server_result[MLMessage.TRAINING_NUM_IN_ROUND]
        sample_num_dict = server_result[MLMessage.SAMPLE_NUM_DICT]
        local_agg_client_result = {}
        local_sample_num_dict = {}
        client_runtime_info = {}
        self.hierarchical_aggregator.reset()
        for client_index in client_indexes:
            # logging.info(
            #     "#######training########### Simulating client_index = %d"
            #     % (client_index)
            # )
            start_time = time.time()

            self.trainer.update_trainer(int(client_index), server_result)
            self.trainer.update_dataset(int(client_index))
            self.args.round_idx += 1
            # weights, local_sample_num = self.trainer.train(self.args.round_idx)
            client_result, local_sample_num = self.trainer.train(self.args.round_idx)
            self.hierarchical_aggregator.local_aggregate_seq(client_index, client_result, local_sample_num, training_num_in_round)
            local_sample_num_dict[client_index] = local_sample_num

            if hasattr(self.args, "simulation_gpu_hetero") and self.args.simulation_gpu_hetero:
                t_train = time.time() - start_time
                logging.info(f"Simulating simulation_gpu_hetero:{runtime_speed_ratio_gpu}, sleep time: {t_train}")
                time.sleep(runtime_speed_ratio_gpu * t_train)

            if hasattr(self.args, "simulation_environment_hetero") and self.args.simulation_environment_hetero:
                t_train = time.time() - start_time
                logging.info(f"Simulating simulation_environment_hetero:{runtime_speed_ratio_env}, sleep time: {t_train}")
                time.sleep(runtime_speed_ratio_env * t_train)

            end_time = time.time()
            client_runtime = end_time - start_time
            client_runtime_info[client_index] = client_runtime
            # logging.info(
            #     "#######training########### End Simulating client_index = %d, consuming time: %f"
            #     % (client_index, client_runtime)
            # )
        if len(client_indexes) == 0:
            local_agg_client_result = {}
        else:
            local_agg_client_result = self.hierarchical_aggregator.end_local_aggregate_seq()
        self.send_local_agg_result_to_server(0, local_agg_client_result,
                local_sample_num_dict, client_runtime_info)








