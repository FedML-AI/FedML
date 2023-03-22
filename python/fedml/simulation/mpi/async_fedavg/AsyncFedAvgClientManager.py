import logging
import time


from .message_define import MyMessage
from .utils import transform_list_to_tensor
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message

from ....utils.model_utils import get_name_params_difference


class AsyncFedAVGClientManager(FedMLCommManager):
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
        self.round_idx = 0
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
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.round_idx = 0
        self.__train(global_model_params, client_index)

    def start_training(self):
        self.round_idx = 0
        # self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.round_idx += 1
        self.__train(global_model_params, client_index)
        if self.round_idx == self.num_rounds - 1:
            # post_complete_message_to_sweep_process(self.args)
            self.finish()


    def send_result_to_server(self, receive_id, weights, local_sample_num, client_runtime_info):
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_RUNTIME_INFO, client_runtime_info)
        self.send_message(message)


    def __train(self, global_model_params, client_index):
        logging.info("#######training########### round_id = %d" % self.round_idx)

        local_agg_model_params = {}
        client_runtime_info = {}
        logging.info("#######training########### Simulating client_index = %d" % (client_index))
        start_time = time.time()
        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index))
        weights, local_sample_num = self.trainer.train(self.round_idx)

        end_time = time.time()
        client_runtime = end_time - start_time
        client_runtime_info[client_index] = client_runtime
        logging.info("#######training########### End Simulating client_index = %d, consuming time: %f" % \
            (client_index, client_runtime))

        # diff_weights = get_name_params_difference(global_model_params, weights) # weights - global_model_params
        self.send_result_to_server(0, weights, local_sample_num, client_runtime_info)











