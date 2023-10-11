from .message_define import MyMessage
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message

class HostManager(FedMLCommManager):
    def __init__(self, args, comm, rank, size, trainer):
        """
        Initialize a HostManager instance.

        Args:
            args: Configuration arguments.
            comm: MPI communication object.
            rank: Rank of the process.
            size: Number of processes in the communicator.
            trainer: Trainer for host-specific tasks.
        """
        super().__init__(args, comm, rank, size)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

    def run(self):
        """Start the HostManager."""
        super().run()

    def register_message_receive_handlers(self):
        """Register message receive handlers."""
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_GRADIENT,
            self.handle_message_receive_gradient_from_server,
        )

    def handle_message_init(self, msg_params):
        """
        Handle the initialization message.

        Args:
            msg_params: Parameters from the initialization message.
        """
        self.round_idx = 0
        self.__train()

    def handle_message_receive_gradient_from_server(self, msg_params):
        """
        Handle the gradient message received from the server.

        Args:
            msg_params: Parameters from the gradient message.
        """
        gradient = msg_params.get(MyMessage.MSG_ARG_KEY_GRADIENT)
        self.trainer.update_model(gradient)
        self.round_idx += 1
        self.__train()
        if self.round_idx == self.num_rounds * self.trainer.get_batch_num():
            self.finish()

    def send_model_to_server(self, receive_id, host_train_logits, host_test_logits):
        """
        Send host training and test logits to the server.

        Args:
            receive_id: ID of the receiver.
            host_train_logits: Host's training logits.
            host_test_logits: Host's test logits.
        """
        message = Message(
            MyMessage.MSG_TYPE_C2S_LOGITS, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_TRAIN_LOGITS, host_train_logits)
        message.add_params(MyMessage.MSG_ARG_KEY_TEST_LOGITS, host_test_logits)
        self.send_message(message)

    def __train(self):
        """Perform host training and send logits to the server."""
        host_train_logits, host_test_logits = self.trainer.computer_logits(
            self.round_idx
        )
        self.send_model_to_server(0, host_train_logits, host_test_logits)
