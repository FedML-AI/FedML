import logging

from .message_define import MyMessage
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class SplitNNClientManager(FedMLCommManager):
    """
    Manages the client-side operations for Split Learning in a Federated Learning setting.

    Args:
        arg_dict (dict): A dictionary containing necessary arguments.
        trainer (Trainer): The trainer responsible for the client's model.
        backend (str): The communication backend (e.g., "MPI").

    Attributes:
        trainer (Trainer): The trainer responsible for the client's model.
        args (args): Arguments for the client manager.
    """
    def __init__(self, arg_dict, trainer, backend="MPI"):
        """
        Initialize a SplitNNClientManager.

        Args:
            arg_dict (dict): A dictionary containing necessary arguments.
            trainer (Trainer): The trainer responsible for the client's model.
            backend (str): The communication backend (e.g., "MPI").
        """
        super().__init__(
            arg_dict["args"],
            arg_dict["comm"],
            arg_dict["rank"],
            arg_dict["max_rank"] + 1,
            backend,
        )
        self.trainer = trainer
        self.trainer.train_mode()
        self.args.round_idx = 0

    def run(self):
        """
        Start the client manager.

        If the trainer's rank is 1, it starts the protocol by running the forward pass.
        """
        if self.trainer.rank == 1:
            logging.info("Starting protocol from rank 1 process")
            self.run_forward_pass()
        super().run()

    def register_message_receive_handlers(self):
        """
        Register message receive handlers for different message types.
        """
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2C_SEMAPHORE, self.handle_message_semaphore
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_GRADS, self.handle_message_gradients
        )

    def handle_message_semaphore(self, msg_params):
        """
        Handle the semaphore message and start the training process.

        Args:
            msg_params: Parameters of the semaphore message.
        """
        # no point in checking the semaphore message
        logging.info("Starting training at node {}".format(self.trainer.rank))
        self.trainer.train_mode()
        self.run_forward_pass()

    def run_forward_pass(self):
        """
        Run the forward pass of the trainer.

        Sends activations and labels to the server afterward.
        """
        acts, labels = self.trainer.forward_pass()
        self.send_activations_and_labels_to_server(
            acts, labels, self.trainer.SERVER_RANK
        )
        self.trainer.batch_idx += 1

    def run_eval(self):
        """
        Run the evaluation process for the client.

        This method sends a validation signal to the server, switches the trainer to evaluation mode,
        and performs the forward pass for validation data. After validation, it sends a validation
        completion signal to the server and updates the round index. If the maximum number of
        epochs per node is reached, it sends a finish signal to the server.

        """
        self.send_validation_signal_to_server(self.trainer.SERVER_RANK)
        self.trainer.eval_mode()
        for i in range(len(self.trainer.testloader)):
            self.run_forward_pass()
        self.send_validation_over_to_server(self.trainer.SERVER_RANK)
        self.args.round_idx += 1
        if (
            self.args.round_idx == self.trainer.MAX_EPOCH_PER_NODE
            and self.trainer.rank == self.trainer.MAX_RANK
        ):
            self.send_finish_to_server(self.trainer.SERVER_RANK)
        else:
            logging.info(
                "sending semaphore from {} to {}".format(
                    self.trainer.rank, self.trainer.node_right
                )
            )
            self.send_semaphore_to_client(self.trainer.node_right)

        if self.args.round_idx == self.trainer.MAX_EPOCH_PER_NODE:
            self.finish()

    def handle_message_gradients(self, msg_params):
        """
        Handle received gradients and initiate backward pass.

        Args:
            msg_params: Parameters of the received gradients message.
        """
        grads = msg_params.get(MyMessage.MSG_ARG_KEY_GRADS)
        self.trainer.backward_pass(grads)
        if self.trainer.batch_idx == len(self.trainer.trainloader):
            logging.info("Epoch over at node {}".format(self.rank))
            self.args.round_idx += 1
            self.run_eval()
        else:
            self.run_forward_pass()

    def send_activations_and_labels_to_server(self, acts, labels, receive_id):
        """
        Send activations and labels to the server.

        Args:
            acts: Activations to be sent.
            labels: Labels corresponding to the activations.
            receive_id: ID of the receiving entity (typically, the server).
        """
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_ACTS, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_ACTS, (acts, labels))
        self.send_message(message)

    def send_semaphore_to_client(self, receive_id):
        """
        Send a semaphore message to a client.

        Args:
            receive_id: ID of the receiving client.
        """
        message = Message(
            MyMessage.MSG_TYPE_C2C_SEMAPHORE, self.get_sender_id(), receive_id
        )
        self.send_message(message)

    def send_validation_signal_to_server(self, receive_id):
        """
        Send a validation signal message to the server.

        Args:
            receive_id: ID of the receiving entity (typically, the server).
        """
        message = Message(
            MyMessage.MSG_TYPE_C2S_VALIDATION_MODE, self.get_sender_id(), receive_id
        )
        self.send_message(message)

    def send_validation_over_to_server(self, receive_id):
        """
        Send a validation completion signal message to the server.

        Args:
            receive_id: ID of the receiving entity (typically, the server).
        """
        message = Message(
            MyMessage.MSG_TYPE_C2S_VALIDATION_OVER, self.get_sender_id(), receive_id
        )
        self.send_message(message)

    def send_finish_to_server(self, receive_id):
        """
        Send a finish signal message to the server.

        Args:
            receive_id: ID of the receiving entity (typically, the server).
        """
        message = Message(
            MyMessage.MSG_TYPE_C2S_PROTOCOL_FINISHED, self.get_sender_id(), receive_id
        )
        self.send_message(message)
