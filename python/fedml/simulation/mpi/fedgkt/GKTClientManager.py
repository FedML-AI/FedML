import logging

from .message_def import MyMessage
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class GKTClientManager(FedMLCommManager):
    """
    A class representing the client-side manager for Global Knowledge Transfer (GKT).

    This manager is responsible for coordinating communication between the client and the server
    during the GKT training process.

    Args:
        args (argparse.Namespace): Additional arguments and settings.
        trainer (GKTClientTrainer): The client-side trainer responsible for training the client model.
        comm (MPI.Comm): MPI communication object.
        rank (int): The rank or identifier of the client process.
        size (int): The total number of processes in the communication group.
        backend (str): The MPI backend for communication (default is "MPI").

    Attributes:
        args (argparse.Namespace): Additional arguments and settings.
        trainer (GKTClientTrainer): The client-side trainer responsible for training the client model.
        num_rounds (int): The total number of communication rounds.
        device (torch.device): The device (e.g., GPU) used for training.
        args.round_idx (int): The current round index.

    Methods:
        run(): Start the client manager to initiate communication and training.
        register_message_receive_handlers(): Register message receive handlers for communication.
        handle_message_init(msg_params): Handle the initialization message from the server.
        handle_message_receive_logits_from_server(msg_params): Handle logits received from the server.
        send_model_to_server(extracted_feature_dict, logits_dict, labels_dict, extracted_feature_dict_test, labels_dict_test):
            Send extracted features, logits, and labels to the server for knowledge transfer.
        __train(): Start the client model training process.

    """
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        """
        Initialize the GKT (Global Knowledge Transfer) client manager.

        Args:
            args: Additional arguments and settings.
            trainer: The GKT client trainer instance.
            comm: The MPI communication object.
            rank: The rank of the current process.
            size: The total number of processes.
            backend: The communication backend (default: "MPI").

        Returns:
            None
        """
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.args.round_idx = 0

    def run(self):
        """
        Start the GKT client manager.

        Args:
            None

        Returns:
            None
        """
        super().run()

    def register_message_receive_handlers(self):
        """
        Register message receive handlers for the GKT client manager.

        Args:
            None

        Returns:
            None
        """
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_TO_CLIENT,
            self.handle_message_receive_logits_from_server,
        )

    def handle_message_init(self, msg_params):
        """
        Handle the initialization message from the server.

        Args:
            msg_params: Parameters from the received message.

        Returns:
            None
        """
        logging.info("handle_message_init. Rank = " + str(self.rank))
        self.args.round_idx = 0
        self.__train()

    def handle_message_receive_logits_from_server(self, msg_params):
        """
        Handle the message containing logits from the server.

        Args:
            msg_params: Parameters from the received message.

        Returns:
            None
        """
        logging.info(
            "handle_message_receive_logits_from_server. Rank = " + str(self.rank)
        )
        global_logits = msg_params.get(MyMessage.MSG_ARG_KEY_GLOBAL_LOGITS)
        self.trainer.update_large_model_logits(global_logits)
        self.args.round_idx += 1
        self.__train()
        if self.args.round_idx == self.num_rounds - 1:
            self.finish()

    def send_model_to_server(
        self,
        receive_id,
        extracted_feature_dict,
        logits_dict,
        labels_dict,
        extracted_feature_dict_test,
        labels_dict_test,
    ):
        """
        Send extracted features, logits, and labels to the server.

        Args:
            receive_id: The ID of the recipient (usually the server).
            extracted_feature_dict: A dictionary of extracted features.
            logits_dict: A dictionary of logits.
            labels_dict: A dictionary of labels.
            extracted_feature_dict_test: A dictionary of extracted features for testing.
            labels_dict_test: A dictionary of labels for testing.

        Returns:
            None
        """
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_FEATURE_AND_LOGITS,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_FEATURE, extracted_feature_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_LOGITS, logits_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_LABELS, labels_dict)
        message.add_params(
            MyMessage.MSG_ARG_KEY_FEATURE_TEST, extracted_feature_dict_test
        )
        message.add_params(MyMessage.MSG_ARG_KEY_LABELS_TEST, labels_dict_test)
        self.send_message(message)

    def __train(self):
        """
        Perform the training process for the GKT client.

        Args:
            None

        Returns:
            None
        """
        logging.info("#######training########### round_id = %d" % self.args.round_idx)
        (
            extracted_feature_dict,
            logits_dict,
            labels_dict,
            extracted_feature_dict_test,
            labels_dict_test,
        ) = self.trainer.train()
        logging.info("#################finish training##############################")
        self.send_model_to_server(
            0,
            extracted_feature_dict,
            logits_dict,
            labels_dict,
            extracted_feature_dict_test,
            labels_dict_test,
        )
