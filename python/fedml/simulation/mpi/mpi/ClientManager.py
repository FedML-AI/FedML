import logging
import time
from math import cos

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
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.trainer.update_trainer(int(client_index), server_result)
        self.trainer.update_dataset(int(client_index))
        self.args.round_idx = 0
        self.__train()

    def start_training(self):
        self.args.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        # model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        server_result = msg_params.get(MyMessage.MSG_ARG_KEY_SERVER_RESULT)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.trainer.update_trainer(int(client_index), server_result)
        self.trainer.update_dataset(int(client_index))
        self.args.round_idx += 1
        self.__train()
        if self.args.round_idx == self.num_rounds - 1:
            # post_complete_message_to_sweep_process(self.args)
            # self.finish()
            pass

    def send_model_to_server(self, receive_id, client_result, local_sample_num):
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.get_sender_id(),
            receive_id,
        )
        # message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_RESULT, client_result)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def __train(self):
        logging.info("#######training########### round_id = %d" % self.args.round_idx)
        if hasattr(self.args, "simulation_gpu_hetero") and self.args.simulation_gpu_hetero:
            # runtime_speed_ratio
            # runtime_speed_ratio * t_train - t_train
            # time.sleep(runtime_speed_ratio * t_train - t_train)
            simulation_gpu_hetero = self.args.simulation_gpu_hetero
            runtime_speed_ratio_gpu = self.args.gpu_hetero_ratio * self.worker_id / self.args.worker_num

        if hasattr(self.args, "simulation_environment_hetero"):
            # runtime_speed_ratio
            # runtime_speed_ratio * t_train - t_train
            # time.sleep(runtime_speed_ratio * t_train - t_train)
            if self.args.simulation_environment_hetero == "cos":
                runtime_speed_ratio_env = self.args.environment_hetero_ratio * \
                    (1 + cos(self.args.round_idx / self.num_rounds*3.1415926 + self.worker_id))
            else:
                raise NotImplementedError

        start_time = time.time()
        # weights, local_sample_num = self.trainer.train(self.args.round_idx)
        client_result, local_sample_num = self.trainer.train(self.args.round_idx)
        if hasattr(self.args, "simulation_gpu_hetero") and self.args.simulation_gpu_hetero:
            t_train = time.time() - start_time
            logging.info(f"Simulating simulation_gpu_hetero:{runtime_speed_ratio_gpu}, sleep time: {t_train}")
            time.sleep(runtime_speed_ratio_gpu * t_train)

        if hasattr(self.args, "simulation_environment_hetero") and self.args.simulation_environment_hetero:
            t_train = time.time() - start_time
            logging.info(f"Simulating simulation_environment_hetero:{runtime_speed_ratio_env}, sleep time: {t_train}")
            time.sleep(runtime_speed_ratio_env * t_train)

        self.send_model_to_server(0, client_result, local_sample_num)
