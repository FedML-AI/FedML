import logging

from fedml_api.distributed.fedgkt.message_def import MyMessage
from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message


class GKTClientMananger(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)

        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_TO_CLIENT,
                                              self.handle_message_receive_logits_from_server)

    def handle_message_init(self, msg_params):
        logging.info("handle_message_init. Rank = " + str(self.rank))
        self.round_idx = 0
        self.__train()

    def handle_message_receive_logits_from_server(self, msg_params):
        logging.info("handle_message_receive_logits_from_server. Rank = " + str(self.rank))
        global_logits = msg_params.get(MyMessage.MSG_ARG_KEY_GLOBAL_LOGITS)
        self.trainer.update_large_model_logits(global_logits)
        self.round_idx += 1
        self.__train()
        if self.round_idx == self.num_rounds - 1:
            self.finish()

    def send_model_to_server(self, receive_id, extracted_feature_dict, logits_dict, labels_dict,
                             extracted_feature_dict_test, labels_dict_test):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_FEATURE_AND_LOGITS, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_FEATURE, extracted_feature_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_LOGITS, logits_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_LABELS, labels_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_FEATURE_TEST, extracted_feature_dict_test)
        message.add_params(MyMessage.MSG_ARG_KEY_LABELS_TEST, labels_dict_test)
        self.send_message(message)

    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        extracted_feature_dict, logits_dict, labels_dict, extracted_feature_dict_test, labels_dict_test = self.trainer.train()
        logging.info("#################finish training##############################")
        self.send_model_to_server(0, extracted_feature_dict, logits_dict, labels_dict, extracted_feature_dict_test, labels_dict_test)

