import logging

from fedml_api.distributed.fedgkt.message_def import MyMessage
from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.server.server_manager import ServerManager


class GKTServerMananger(ServerManager):
    def __init__(self, args, server_trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)

        self.server_trainer = server_trainer
        self.round_num = args.comm_round
        self.round_idx = 0

        self.count = 0

    def run(self):
        global_model_params = None
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params)
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_FEATURE_AND_LOGITS,
                                              self.handle_message_receive_feature_and_logits_from_client)

    def handle_message_receive_feature_and_logits_from_client(self, msg_params):
        logging.info("handle_message_receive_feature_and_logits_from_client")
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        extracted_feature_dict = msg_params.get(MyMessage.MSG_ARG_KEY_FEATURE)
        logits_dict = msg_params.get(MyMessage.MSG_ARG_KEY_LOGITS)
        labels_dict = msg_params.get(MyMessage.MSG_ARG_KEY_LABELS)
        extracted_feature_dict_test = msg_params.get(MyMessage.MSG_ARG_KEY_FEATURE_TEST)
        labels_dict_test = msg_params.get(MyMessage.MSG_ARG_KEY_LABELS_TEST)

        self.server_trainer.add_local_trained_result(sender_id - 1, extracted_feature_dict, logits_dict, labels_dict,
                                                 extracted_feature_dict_test, labels_dict_test)
        b_all_received = self.server_trainer.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            self.server_trainer.train(self.round_idx)

            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                self.finish()
                return

            for receiver_id in range(1, self.size):
                global_logits = self.server_trainer.get_global_logits(receiver_id-1)
                self.send_message_sync_model_to_client(receiver_id, global_logits)

    def send_message_init_config(self, receive_id, global_model_params):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        self.send_message(message)
        logging.info("send_message_init_config. Receive_id: " + str(receive_id))

    def send_message_sync_model_to_client(self, receive_id, global_logits):
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_GLOBAL_LOGITS, global_logits)
        self.send_message(message)
        logging.info("send_message_sync_model_to_client. Receive_id: " + str(receive_id))
