import logging

from .message_define import MyMessage
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class BaseCentralManager(FedMLCommManager):
    def __init__(self, args, comm, rank, size, aggregator):
        super().__init__(args, comm, rank, size)

        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.args.round_idx = 0

    def run(self):
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id)
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_INFORMATION,
            self.handle_message_receive_model_from_client,
        )

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        client_local_result = msg_params.get(MyMessage.MSG_ARG_KEY_INFORMATION)

        self.aggregator.add_client_local_result(sender_id - 1, client_local_result)
        b_all_received = self.aggregator.check_whether_all_receive()

        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            logging.info(
                "**********************************ROUND INDEX = " + str(self.args.round_idx)
            )
            global_result = self.aggregator.aggregate()

            # start the next round
            self.args.round_idx += 1
            if self.args.round_idx == self.round_num:
                self.finish()
                return

            for receiver_id in range(1, self.size):
                self.send_message_to_client(receiver_id, global_result)

    def send_message_init_config(self, receive_id):
        message = Message(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        self.send_message(message)

    def send_message_to_client(self, receive_id, global_result):
        message = Message(
            MyMessage.MSG_TYPE_S2C_INFORMATION, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_INFORMATION, global_result)
        self.send_message(message)
