from .message_define import MyMessage
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class GuestManager(FedMLCommManager):
    """
    Class representing the manager for a guest in a distributed system.

    This class is responsible for handling communication between the guest and other participants,
    as well as coordinating training rounds.

    Attributes:
        args: Arguments for the manager.
        comm: The communication interface.
        rank: The rank of the guest in the communication group.
        size: The total number of participants in the communication group.
        guest_trainer: The trainer responsible for guest-specific training.

    Methods:
        run():
            Start the guest manager and run communication.
        register_message_receive_handlers():
            Register message receive handlers for handling incoming messages.
        handle_message_receive_logits_from_client(msg_params):
            Handle the reception of logits and trigger training when all data is received.
        send_message_init_config(receive_id):
            Send an initialization message to a client.
        send_message_to_client(receive_id, global_result):
            Send a message containing global training results to a client.

    """

    def __init__(self, args, comm, rank, size, guest_trainer):
        """
        Initialize the GuestManager.

        Args:
            args: Arguments for the manager.
            comm: The communication interface.
            rank: The rank of the guest in the communication group.
            size: The total number of participants in the communication group.
            guest_trainer: The trainer responsible for guest-specific training.

        Returns:
            None
        """
        super().__init__(args, comm, rank, size)

        self.guest_trainer = guest_trainer
        self.round_num = args.comm_round
        self.args.round_idx = 0

    def run(self):
        """
        Start the guest manager and run communication.

        Returns:
            None
        """
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id)
        super().run()

    def register_message_receive_handlers(self):
        """
        Register message receive handlers for handling incoming messages.

        Returns:
            None
        """
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_LOGITS,
            self.handle_message_receive_logits_from_client,
        )

    def handle_message_receive_logits_from_client(self, msg_params):
        """
        Handle the reception of logits and trigger training when all data is received.

        Args:
            msg_params: Parameters of the received message.

        Returns:
            None
        """
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
        """
        Send an initialization message to a client.

        Args:
            receive_id: The ID of the client to receive the message.

        Returns:
            None
        """
        message = Message(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        self.send_message(message)

    def send_message_to_client(self, receive_id, global_result):
        """
        Send a message containing global training results to a client.

        Args:
            receive_id: The ID of the client to receive the message.
            global_result: The global training result to send.

        Returns:
            None
        """
        message = Message(
            MyMessage.MSG_TYPE_S2C_GRADIENT, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_GRADIENT, global_result)
        self.send_message(message)
