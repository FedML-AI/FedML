import logging
import numpy as np

from .message_define import MyMessage
from .utils import transform_tensor_to_list
from ....core.distributed.communication.message import Message
from ....core.distributed.fedml_comm_manager import FedMLCommManager


class AsyncFedAVGServerManager(FedMLCommManager):
    def __init__(
        self,
        args,
        aggregator,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
        is_preprocessed=False,
        preprocessed_client_lists=None,
    ):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists
        self.client_round_dict = {}
        for i in range(self.args.client_num_in_total):
            self.client_round_dict[i] = 0


    def run(self):
        super().run()


    def send_init_msg(self):
        # sampling clients
        # client_indexes = self.aggregator.client_sampling(
        #     self.round_idx,
        #     self.args.client_num_in_total,
        #     self.args.client_num_per_round,
        # )
        num_clients = min(self.args.client_num_per_round, self.args.client_num_in_total)
        client_indexes = np.random.choice(
            range(self.args.client_num_in_total), num_clients, replace=False
        )
        global_model_params = self.aggregator.get_global_model_params()
        for process_id in range(1, self.size):
            self.send_message_init_config(
                process_id, global_model_params, client_indexes[process_id - 1]
            )


    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.handle_message_receive_model_from_client,
        )

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        client_runtime_info = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_RUNTIME_INFO)
        self.aggregator.record_client_runtime(sender_id - 1, client_runtime_info)

        # start the next round
        self.round_idx += 1
        self.aggregator.add_local_trained_result(
            sender_id - 1, model_params, local_sample_number,
            current_round=self.round_idx, client_round=self.client_round_dict[sender_id - 1]
        )
        self.aggregator.test_on_server_for_all_clients(self.round_idx)

        if self.round_idx == self.round_num:
            # post_complete_message_to_sweep_process(self.args)
            self.finish()
            print("here")
            return
        if self.is_preprocessed:
            if self.preprocessed_client_lists is None:
                # sampling has already been done in data preprocessor
                client_indexes = [self.round_idx] * self.args.client_num_per_round
            else:
                client_indexes = self.preprocessed_client_lists[self.round_idx]
        else:
            # sampling clients
            client_indexes = self.aggregator.client_sampling(
                self.round_idx,
                self.args.client_num_in_total,
                self.args.client_num_per_round,
            )

        global_model_params = self.aggregator.get_global_model_params()

        print("indexes of clients: " + str(client_indexes))
        print("size = %d" % self.size)

        self.send_message_sync_model_to_client(
            sender_id, global_model_params, 
            client_indexes
        )
        self.client_round_dict[client_indexes[0]] = self.round_idx


    def send_message_init_config(self, receive_id, global_model_params, 
                                client_index):
        message = Message(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)


    def send_message_sync_model_to_client(self, receive_id, global_model_params, 
                                client_index):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)
