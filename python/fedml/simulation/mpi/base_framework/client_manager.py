from .message_define import MyMessage
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class BaseClientManager(FedMLCommManager):
    """
    Base class representing a client manager in a distributed system.

    This class handles the communication between clients and the central server.

    Attributes:
        args (object): An object containing configuration parameters.
        comm (object): A communication object for MPI communication.
        rank (int): The rank of the current process.
        size (int): The total number of processes.
        trainer (object): An object responsible for client-side training.
        num_rounds (int): The total number of communication rounds.

    Methods:
        run():
            Start the client manager.
        handle_message_init(msg_params):
            Handle initialization message from the server.
        handle_message_receive_model_from_server(msg_params):
            Handle receiving model update from the server.
        send_model_to_server(receive_id, client_gradient):
            Send client-side model updates to the server.
        __train():
            Perform training and send updates to the server.
    """
    def __init__(self, args, comm, rank, size, trainer):
        """
        Initialize the BaseClientManager.

        Args:
            args (object): An object containing configuration parameters.
            comm (object): A communication object for MPI communication.
            rank (int): The rank of the current process.
            size (int): The total number of processes.
            trainer (object): An object responsible for client-side training.

        Returns:
            None
        """
        super().__init__(args, comm, rank, size)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.args.round_idx = 0

    def run(self):
        """
        Start the client manager.

        Args:
            None

        Returns:
            None
        """
        super().run()

    def register_message_receive_handlers(self):
        """
        Register message receive handlers for different message types.

        Args:
            None

        Returns:
            None
        """
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_INFORMATION,
            self.handle_message_receive_model_from_server,
        )

    def handle_message_init(self, msg_params):
        """
        Handle initialization message from the server.

        Args:
            msg_params (dict): Parameters included in the message.

        Returns:
            None
        """
        self.trainer.update(0)
        self.args.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        """
        Handle receiving model update from the server.

        Args:
            msg_params (dict): Parameters included in the message.

        Returns:
            None
        """
        global_result = msg_params.get(MyMessage.MSG_ARG_KEY_INFORMATION)
        self.trainer.update(global_result)
        self.args.round_idx += 1
        self.__train()
        if self.args.round_idx == self.num_rounds - 1:
            self.finish()

    def send_model_to_server(self, receive_id, client_gradient):
        """
        Send client-side model updates to the server.

        Args:
            receive_id (int): The ID of the recipient (server).
            client_gradient (object): The client-side model update.

        Returns:
            None
        """
        message = Message(
            MyMessage.MSG_TYPE_C2S_INFORMATION, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_INFORMATION, client_gradient)
        self.send_message(message)

    def __train(self):
        """
        Perform training and send updates to the server.

        Args:
            None

        Returns:
            None
        """
        # do something here (e.g., training)
        training_interation_result = self.trainer.train()

        # send something calculated to the server side (we use client_gradient = 1 as an example)
        self.send_model_to_server(0, training_interation_result)
