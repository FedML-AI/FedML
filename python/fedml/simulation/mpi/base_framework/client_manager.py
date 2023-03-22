from .message_define import MyMessage
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class BaseClientManager(FedMLCommManager):
    def __init__(self, args, comm, rank, size, trainer):
        super().__init__(args, comm, rank, size)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.args.round_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_INFORMATION,
            self.handle_message_receive_model_from_server,
        )

    def handle_message_init(self, msg_params):
        self.trainer.update(0)
        self.args.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        global_result = msg_params.get(MyMessage.MSG_ARG_KEY_INFORMATION)
        self.trainer.update(global_result)
        self.args.round_idx += 1
        self.__train()
        if self.args.round_idx == self.num_rounds - 1:
            self.finish()

    def send_model_to_server(self, receive_id, client_gradient):
        message = Message(
            MyMessage.MSG_TYPE_C2S_INFORMATION, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_INFORMATION, client_gradient)
        self.send_message(message)

    def __train(self):
        # do something here (e.g., training)
        training_interation_result = self.trainer.train()

        # send something calculated to the server side (we use client_gradient = 1 as an example)
        self.send_model_to_server(0, training_interation_result)
