import logging

from .message_define import MyMessage
from .utils import transform_tensor_to_list
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class FedGANServerManager(FedMLCommManager):
    """
    Manager for Federated GAN server-side operations.

    Args:
        args: Configuration arguments.
        aggregator: Aggregator for model updates.
        comm: MPI communication object.
        rank (int): Rank of the current process.
        size (int): Total number of processes.
        backend (str): Backend for communication (e.g., 'MPI').
        is_preprocessed (bool): Indicates if client sampling is preprocessed.
        preprocessed_client_lists (list): Preprocessed client sampling lists.

    Attributes:
        args: Configuration arguments.
        aggregator: Aggregator for model updates.
        round_num: Number of communication rounds.
        args.round_idx: Current communication round index.
        is_preprocessed: Indicates if client sampling is preprocessed.
        preprocessed_client_lists: Preprocessed client sampling lists.
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
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.args.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists

    def run(self):
        """
        Start the server manager's execution.
        """
        super().run()

    def send_init_msg(self):
        """
        Send initialization message to clients, including global model parameters and client indexes.
        """
        # Sampling clients
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
        Register message receive handlers for receiving model updates from clients.
        """
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.handle_message_receive_model_from_client,
        )

    def handle_message_receive_model_from_client(self, msg_params):
        """
        Handle the received model update message from a client.

        Args:
            msg_params (dict): Message parameters containing sender ID, model parameters, and local sample count.
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
            # self.aggregator.test_on_server_for_all_clients(self.args.round_idx)

            # Start the next round
            self.args.round_idx += 1
            if self.args.round_idx == self.round_num:
                # post_complete_message_to_sweep_process(self.args)
                self.finish()
                print("here")
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
        Send initialization configuration message to a client.

        Args:
            receive_id (int): ID of the client receiving the configuration.
            global_model_params: Global model parameters.
            client_index: Index of the client.
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
            receive_id (int): ID of the client receiving the model.
            global_model_params: Global model parameters.
            client_index: Index of the client.
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
