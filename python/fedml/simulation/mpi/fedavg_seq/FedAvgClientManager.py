import logging
import time
from math import cos

from .message_define import MyMessage
from .utils import transform_list_to_tensor
from ....core.distributed.communication.message import Message
from ....core.distributed.fedml_comm_manager import FedMLCommManager


class FedAVGClientManager(FedMLCommManager):
    def __init__(
        self, args, trainer, comm=None, rank=0, size=0, backend="MPI",
    ):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.worker_id = self.rank - 1

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init)
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.handle_message_receive_model_from_server,
        )

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        # client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        average_weight_dict = msg_params.get(MyMessage.MSG_ARG_KEY_AVG_WEIGHTS)
        client_schedule = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_SCHEDULE)
        client_indexes = client_schedule[self.worker_id]

        self.round_idx = 0
        self.__train(global_model_params, client_indexes, average_weight_dict)

    def start_training(self):
        self.round_idx = 0
        # self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        # client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        average_weight_dict = msg_params.get(MyMessage.MSG_ARG_KEY_AVG_WEIGHTS)
        client_schedule = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_SCHEDULE)
        client_indexes = client_schedule[self.worker_id]

        self.round_idx += 1
        self.__train(global_model_params, client_indexes, average_weight_dict)
        if self.round_idx == self.num_rounds - 1:
            # post_complete_message_to_sweep_process(self.args)
            self.finish()

    def send_result_to_server(self, receive_id, weights, client_runtime_info):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id,)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        # message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_RUNTIME_INFO, client_runtime_info)
        self.send_message(message)

    def add_client_model(self, local_agg_model_params, model_params, weight=1.0):
        # Add params that needed to be reduces from clients
        for name, param in model_params.items():
            if name not in local_agg_model_params:
                local_agg_model_params[name] = param * weight
            else:
                local_agg_model_params[name] += param * weight

    def __train(self, global_model_params, client_indexes, average_weight_dict):
        logging.info("#######training########### round_id = %d" % self.round_idx)

        if hasattr(self.args, "simulation_gpu_hetero"):
            # runtime_speed_ratio
            # runtime_speed_ratio * t_train - t_train
            # time.sleep(runtime_speed_ratio * t_train - t_train)
            simulation_gpu_hetero = self.args.simulation_gpu_hetero
            runtime_speed_ratio = self.args.gpu_hetero_ratio * self.worker_id / self.args.worker_num

        if hasattr(self.args, "simulation_environment_hetero"):
            # runtime_speed_ratio
            # runtime_speed_ratio * t_train - t_train
            # time.sleep(runtime_speed_ratio * t_train - t_train)
            if self.args.simulation_environment_hetero == "cos":
                runtime_speed_ratio = self.args.environment_hetero_ratio * \
                    (1 + cos(self.round_idx / self.num_rounds*3.1415926 + self.worker_id))
            else:
                raise NotImplementedError


        local_agg_model_params = {}
        client_runtime_info = {}
        for client_index in client_indexes:
            logging.info(
                "#######training########### Simulating client_index = %d, average weight: %f "
                % (client_index, average_weight_dict[client_index])
            )
            start_time = time.time()
            self.trainer.update_model(global_model_params)
            self.trainer.update_dataset(int(client_index))
            weights, local_sample_num = self.trainer.train(self.round_idx)
            self.add_client_model(local_agg_model_params, weights, weight=average_weight_dict[client_index])
            if hasattr(self.args, "simulation_gpu_hetero"):
                t_train = time.time() - start_time
                time.sleep(runtime_speed_ratio * t_train)
            end_time = time.time()
            client_runtime = end_time - start_time
            client_runtime_info[client_index] = client_runtime
            logging.info(
                "#######training########### End Simulating client_index = %d, consuming time: %f"
                % (client_index, client_runtime)
            )
        self.send_result_to_server(0, local_agg_model_params, client_runtime_info)
