import logging

from .message_define import MyMessage
from .utils import transform_list_to_tensor
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class FedAVGClientManager(FedMLCommManager):
    """
    Class representing the client manager in the FedAVG federated learning process. 
    """
    def __init__(
        self,
        args,
        trainer,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
    ):
        """
        Initialize the client manager for the FedAVG federated learning process.

        Args:
            args (Namespace): Command-line arguments and configuration for the FedAVG process.
            trainer: The federated learning trainer responsible for local training.
            comm: The communication backend for inter-process communication.
            rank (int): The rank or identifier of the current client.
            size (int): The total number of clients.
            backend (str): The backend for distributed computing (e.g., "MPI").
        """
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.args.round_idx = 0

    def run(self):
        """
        Start the client manager to handle federated learning tasks.
        """
        super().run()

    def register_message_receive_handlers(self):
        """
        Register message handlers for processing incoming messages.
        """
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.handle_message_receive_model_from_server,
        )

    def handle_message_init(self, msg_params):
        """
        Handle the initialization message from the server.

        Args:
            msg_params (dict): Parameters received in the message.
        """
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index))
        self.args.round_idx = 0
        self.__train()

    def start_training(self):
        """
        Start the federated training process.
        """
        self.args.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        """
        Handle the model update message received from the server.

        Args:
            msg_params (dict): Parameters received in the message.
        """
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.trainer.update_model(model_params)
        self.trainer.update_dataset(int(client_index))
        self.args.round_idx += 1
        self.__train()

        if self.args.round_idx == self.num_rounds - 1:
            self.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        """
        Send the locally trained model to the server.

        Args:
            receive_id (int): The ID of the server to receive the model.
            weights: The model parameters.
            local_sample_num (int): The number of local training samples.
        """
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def __train(self):
        """
        Perform federated training for a round.
        """
        logging.info("#######training########### round_id = %d" % self.args.round_idx)
        weights, local_sample_num = self.trainer.train(self.args.round_idx)
        self.send_model_to_server(0, weights, local_sample_num)
