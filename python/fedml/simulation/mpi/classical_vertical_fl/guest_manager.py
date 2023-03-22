from .message_define import MyMessage
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class GuestManager(FedMLCommManager):
    def __init__(self, args, comm, rank, size, guest_trainer):
        super().__init__(args, comm, rank, size)

        self.guest_trainer = guest_trainer
        self.round_num = args.comm_round
        self.args.round_idx = 0

    def run(self):
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id)
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_LOGITS,
            self.handle_message_receive_logits_from_client,
        )

    def handle_message_receive_logits_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        host_train_logits = msg_params.get(MyMessage.MSG_ARG_KEY_TRAIN_LOGITS)
        host_test_logits = msg_params.get(MyMessage.MSG_ARG_KEY_TEST_LOGITS)

        self.guest_trainer.add_client_local_result(
            sender_id - 1, host_train_logits, host_test_logits
        )
        b_all_received = self.guest_trainer.check_whether_all_receive()

        if b_all_received:
            host_gradient = self.guest_trainer.train(self.args.round_idx)

            for receiver_id in range(1, self.size):
                self.send_message_to_client(receiver_id, host_gradient)

            # start the next round
            self.args.round_idx += 1
            if self.args.round_idx == self.round_num * self.guest_trainer.get_batch_num():
                self.finish()

    def send_message_init_config(self, receive_id):
        message = Message(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        self.send_message(message)

    def send_message_to_client(self, receive_id, global_result):
        message = Message(
            MyMessage.MSG_TYPE_S2C_GRADIENT, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_GRADIENT, global_result)
        self.send_message(message)
