from .message_define import MyMessage
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class SplitNNServerManager(FedMLCommManager):
    """
    Manager for the SplitNN server that handles communication.
    """
    def __init__(self, arg_dict, trainer, backend="MPI"):
        """
        Initialize the SplitNNServerManager.

        Args:
            arg_dict (dict): A dictionary containing configuration arguments.
            trainer: The trainer instance for the server.
            backend (str): The communication backend to use (default is "MPI").
        """
        super().__init__(
            arg_dict["args"],
            arg_dict["comm"],
            arg_dict["rank"],
            arg_dict["max_rank"] + 1,
            backend,
        )
        self.trainer = trainer
        self.args.round_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        """
        Register message receive handlers for different message types.
        """
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_ACTS, self.handle_message_acts
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_VALIDATION_MODE, self.handle_message_validation_mode
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_VALIDATION_OVER, self.handle_message_validation_over
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED,
            self.handle_message_finish_protocol,
        )

    def send_grads_to_client(self, receive_id, grads):
        """
        Handle a message containing activations.

        Args:
            msg_params (dict): Parameters of the received message.
        """
        message = Message(
            MyMessage.MSG_TYPE_S2C_GRADS, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_GRADS, grads)
        self.send_message(message)

    def handle_message_acts(self, msg_params):
        """
        Handle a message containing activations.

        Args:
            msg_params (dict): Parameters of the received message.
        """
        acts, labels = msg_params.get(MyMessage.MSG_ARG_KEY_ACTS)
        self.trainer.forward_pass(acts, labels)
        if self.trainer.phase == "train":
            grads = self.trainer.backward_pass()
            self.send_grads_to_client(self.trainer.active_node, grads)

    def handle_message_validation_mode(self, msg_params):
        """
        Handle a message indicating validation mode.

        Args:
            msg_params (dict): Parameters of the received message.
        """
        
        self.trainer.eval_mode()

    def handle_message_validation_over(self, msg_params):
        """
        Handle a message indicating the end of validation.

        Args:
            msg_params (dict): Parameters of the received message.
        """

        self.trainer.validation_over()

    def handle_message_finish_protocol(self):
        """
        Handle a message indicating the protocol has finished.

        Args:
            msg_params (dict): Parameters of the received message.
        """
        self.finish()
