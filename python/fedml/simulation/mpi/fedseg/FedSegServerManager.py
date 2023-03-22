import logging

from .message_define import MyMessage
from .utils import transform_tensor_to_list
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class FedSegServerManager(FedMLCommManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.args.round_idx = 0
        logging.info("Initializing Server Manager")

    def run(self):
        super().run()

    def send_init_msg(self):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(
            self.args.round_idx,
            self.args.client_num_in_total,
            self.args.client_num_per_round,
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
        train_eval_metrics = msg_params.get(
            MyMessage.MSG_ARG_KEY_TRAIN_EVALUATION_METRICS
        )
        test_eval_metrics = msg_params.get(
            MyMessage.MSG_ARG_KEY_TEST_EVALUATION_METRICS
        )

        logging.info("Received model from client {0}".format(sender_id - 1))

        self.aggregator.add_local_trained_result(
            sender_id - 1, model_params, local_sample_number
        )
        self.aggregator.add_client_test_result(
            self.args.round_idx, sender_id - 1, train_eval_metrics, test_eval_metrics
        )

        b_all_received = self.aggregator.check_whether_all_receive()

        if b_all_received:
            global_model_params = self.aggregator.aggregate()
            self.aggregator.output_global_acc_and_loss(self.args.round_idx)

            # start the next round
            self.args.round_idx += 1
            if self.args.round_idx == self.round_num:
                self.finish()
                return

            # sampling clients
            client_indexes = self.aggregator.client_sampling(
                self.args.round_idx,
                self.args.client_num_in_total,
                self.args.client_num_per_round,
            )

            for receiver_id in range(1, self.size):
                self.send_message_sync_model_to_client(
                    receiver_id, global_model_params, client_indexes[receiver_id - 1]
                )

    def send_message_init_config(self, receive_id, global_model_params, client_index):
        logging.info("Initial Configurations sent to client {0}".format(client_index))
        message = Message(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_sync_model_to_client(
        self, receive_id, global_model_params, client_index
    ):
        logging.info(
            "send_message_sync_model_to_client. receive_id {0}".format(receive_id)
        )
        message = Message(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)
