import logging

from .message_define import MyMessage
from .utils import transform_list_to_tensor, post_complete_message_to_sweep_process
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class FedOptClientManager(FedMLCommManager):
    """Manages client-side operations for federated optimization.

    This class is responsible for managing client-side operations during federated optimization.
    It handles communication with the server, updates model parameters, and performs training rounds.

    Attributes:
        args: A configuration object containing client parameters.
        trainer: An instance of the federated optimizer trainer.
        comm: The communication backend.
        rank: The rank of the client in the communication group.
        size: The total number of processes in the communication group.
        backend: The communication backend (e.g., "MPI").

    Methods:
        run(): Runs the client manager to participate in federated optimization.
        register_message_receive_handlers(): Registers message handlers for receiving updates from the server.
        handle_message_init(msg_params): Handles initialization messages from the server.
        start_training(): Starts the federated training process.
        handle_message_receive_model_from_server(msg_params): Handles received model updates from the server.
        send_model_to_server(receive_id, weights, local_sample_num): Sends updated model to the server.
        __train(): Performs the training process.

    """

    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.args.round_idx = 0

    def run(self):
        """Runs the client manager to participate in federated optimization."""
        super().run()

    def register_message_receive_handlers(self):
        """Registers message handlers for receiving updates from the server."""
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.handle_message_receive_model_from_server,
        )

    def handle_message_init(self, msg_params):
        """Handles initialization messages from the server."""
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index))
        self.args.round_idx = 0
        self.__train()

    def start_training(self):
        """Starts the federated training process."""
        self.args.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        """Handles received model updates from the server."""
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.trainer.update_model(model_params)
        self.trainer.update_dataset(int(client_index))
        self.args.round_idx += 1
        self.__train()
        if self.args.round_idx == self.num_rounds - 1:
            post_complete_message_to_sweep_process(self.args)
            self.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        """Sends updated model to the server."""
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def __train(self):
        """Performs the training process."""
        logging.info("#######training########### round_id = %d" % self.args.round_idx)
        weights, local_sample_num = self.trainer.train(self.args.round_idx)
        self.send_model_to_server(0, weights, local_sample_num)
