
import logging

from .message_define import MyMessage
from .utils import transform_tensor_to_list
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class FedAVGServerManager(FedMLCommManager):
    """
    A class that manages the server-side operations in a Federated Averaging (FedAVG) framework.

    This class handles the synchronization of model parameters and training progress across multiple clients
    in a federated learning setting using the FedAVG algorithm.

    Args:
        args: An object containing configuration parameters.
        aggregator: An aggregator object responsible for aggregating client updates.
        comm: A communication object for inter-process communication.
        rank: The rank or ID of this process in the communication group.
        size: The total number of processes in the communication group.
        backend: The backend used for communication (e.g., "MPI" or "gloo").
        is_preprocessed: A flag indicating whether the client data is preprocessed.
        preprocessed_client_lists: A list of preprocessed client data.

    Attributes:
        args: An object containing configuration parameters.
        aggregator: An aggregator object responsible for aggregating client updates.
        round_num: The total number of communication rounds.
        is_preprocessed: A flag indicating whether the client data is preprocessed.
        preprocessed_client_lists: A list of preprocessed client data.

    Methods:
        run(): Start the server manager and enter the main execution loop.
        send_init_msg(): Send an initialization message to clients to start the federated learning process.
        register_message_receive_handlers(): Register message handlers for message types.
        handle_message_receive_model_from_client(msg_params): Handle a message received from a client containing model updates.
        send_message_init_config(receive_id, global_model_params, client_index): Send an initialization message to a specific client.
        send_message_sync_model_to_client(receive_id, global_model_params, client_index): Send a model synchronization message to a client.
    """
    
    def __init__(
        self,
        args,
        aggregator,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
        is_preprocessed=False,
        preprocessed_client_lists=None,
    ):
        """
        Initialize the server manager for the FedAVG federated learning process.

        Args:
            args (Namespace): Command-line arguments and configuration for the FedAVG process.
            aggregator: The federated learning aggregator responsible for model aggregation.
            comm: The communication backend for inter-process communication.
            rank (int): The rank or identifier of the current server.
            size (int): The total number of clients and servers.
            backend (str): The backend for distributed computing (e.g., "MPI").
            is_preprocessed (bool): Whether client sampling has been preprocessed.
            preprocessed_client_lists (list): Preprocessed client sampling lists.
        """
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.args.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists

    def run(self):
        """
        Start the server manager to handle federated learning tasks.
        """
        super().run()

    def send_init_msg(self):
        """
        Send initialization messages to clients, including global model parameters and client indexes.
        """
        client_indexes = self.aggregator.client_sampling(
            self.args.round_idx,
            self.args.client_num_in_total,
            self.args.client_num_per_round,
        )
        global_model_params = self.aggregator.get_global_model_params()
        for process_id in range(1, self.size):
            self.send_message_init_config(
                process_id, global_model_params, client_indexes[process_id - 1]
            )

    def register_message_receive_handlers(self):
        """
        Register message handlers for processing incoming messages.
        """
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.handle_message_receive_model_from_client,
        )

    def handle_message_receive_model_from_client(self, msg_params):
        """
        Handle the model update message received from a client.

        Args:
            msg_params (dict): Parameters received in the message.
        """
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(
            sender_id - 1, model_params, local_sample_number
        )
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            global_model_params = self.aggregator.aggregate()
            self.aggregator.test_on_server_for_all_clients(self.args.round_idx)

            # Start the next round
            self.args.round_idx += 1
            if self.args.round_idx == self.round_num:
                self.finish()
                return
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # Sampling has already been done in data preprocessor
                    client_indexes = [self.args.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.args.round_idx]
            else:
                # Sampling clients
                client_indexes = self.aggregator.client_sampling(
                    self.args.round_idx,
                    self.args.client_num_in_total,
                    self.args.client_num_per_round,
                )

            print("indexes of clients: " + str(client_indexes))
            print("size = %d" % self.size)

            for receiver_id in range(1, self.size):
                self.send_message_sync_model_to_client(
                    receiver_id, global_model_params, client_indexes[receiver_id - 1]
                )

    def send_message_init_config(self, receive_id, global_model_params, client_index):
        """
        Send an initialization message to a client.

        Args:
            receive_id (int): The ID of the client to receive the message.
            global_model_params: The global model parameters.
            client_index: The index of the client.
        """
        message = Message(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_sync_model_to_client(
        self, receive_id, global_model_params, client_index
    ):
        """
        Send a model synchronization message to a client.

        Args:
            receive_id (int): The ID of the client to receive the message.
            global_model_params: The global model parameters.
            client_index: The index of the client.
        """
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

        