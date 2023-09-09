import logging
import numpy as np

from .message_define import MyMessage
from .utils import transform_tensor_to_list, post_complete_message_to_sweep_process
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class FedOptServerManager(FedMLCommManager):
    """
    Manager for the Federated Optimization (FedOpt) Server.

    Args:
        args (object): Arguments for configuration.
        aggregator (object): Aggregator for Federated Optimization.
        comm (object, optional): Communication module (default: None).
        rank (int, optional): Server's rank (default: 0).
        size (int, optional): Total number of workers (default: 0).
        backend (str, optional): Backend for communication (default: "MPI").
        is_preprocessed (bool, optional): Flag indicating preprocessed data (default: False).
        preprocessed_client_lists (list, optional): Preprocessed client lists (default: None).

    Attributes:
        args (object): Arguments for configuration.
        aggregator (object): Aggregator for Federated Optimization.
        round_num (int): Number of communication rounds.
        round_idx (int): Current communication round index.
        is_preprocessed (bool): Flag indicating preprocessed data.
        preprocessed_client_lists (list): Preprocessed client lists.
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
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists

    def run(self):
        super().run()

    def send_init_msg(self):
        """
        Send initialization messages to clients.

        Notes:
            This method initializes and sends configuration messages to clients for the
            start of a new communication round.
        """
        # sampling clients
        client_indexes = self.aggregator.client_sampling(
            self.round_idx,
            self.args.client_num_in_total,
            self.args.client_num_per_round,
        )

        client_schedule = self.aggregator.generate_client_schedule(self.round_idx, client_indexes)
        average_weight_dict = self.aggregator.get_average_weight(client_indexes)

        global_model_params = self.aggregator.get_global_model_params()
        for process_id in range(1, self.size):
            self.send_message_init_config(
                process_id, global_model_params, 
                average_weight_dict, client_schedule
            )

    def register_message_receive_handlers(self):
        """
        Register handlers for receiving messages.

        Notes:
            This method registers message handlers for the server to process incoming
            messages from clients.
        """
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.handle_message_receive_model_from_client,
        )

    def handle_message_receive_model_from_client(self, msg_params):
        """
        Handle the received model from a client.

        Args:
            msg_params (dict): Message parameters.

        Notes:
            This method handles the received model from a client, records client
            runtime information, adds local trained results, and checks whether all
            clients have sent their updates to proceed to the next round.
        """
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        # local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        client_runtime_info = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_RUNTIME_INFO)
        self.aggregator.record_client_runtime(sender_id - 1, client_runtime_info)

        self.aggregator.add_local_trained_result(
            sender_id - 1, model_params,
        )
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            global_model_params = self.aggregator.aggregate()
            self.aggregator.test_on_server_for_all_clients(self.round_idx)

            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                post_complete_message_to_sweep_process(self.args)
                self.finish()
                return

            # sampling clients
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.round_idx]
            else:
                # # sampling clients
                client_indexes = self.aggregator.client_sampling(
                    self.round_idx,
                    self.args.client_num_in_total,
                    self.args.client_num_per_round,
                )
            client_schedule = self.aggregator.generate_client_schedule(self.round_idx, client_indexes)
            average_weight_dict = self.aggregator.get_average_weight(client_indexes)

            global_model_params = self.aggregator.get_global_model_params()

            print("size = %d" % self.size)

            for receiver_id in range(1, self.size):
                self.send_message_sync_model_to_client(
                    receiver_id, global_model_params,
                    average_weight_dict, client_schedule
                )

    def send_message_init_config(self, receive_id, global_model_params, 
                                average_weight_dict, client_schedule):
        """
        Send initialization configuration message to a client.

        Args:
            receive_id (int): Receiver's ID.
            global_model_params (dict): Global model parameters.
            average_weight_dict (dict): Dictionary of average weights for clients.
            client_schedule (list): Schedule of clients for the round.

        Notes:
            This method constructs and sends an initialization configuration message to
            a client.
        """
        message = Message(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        # message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_AVG_WEIGHTS, average_weight_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_SCHEDULE, client_schedule)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, 
                                average_weight_dict, client_schedule):
        """
        Send model synchronization message to a client.

        Args:
            receive_id (int): Receiver's ID.
            global_model_params (dict): Global model parameters.
            average_weight_dict (dict): Dictionary of average weights for clients.
            client_schedule (list): Schedule of clients for the round.

        Notes:
            This method constructs and sends a model synchronization message to a client.
        """
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        # message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_AVG_WEIGHTS, average_weight_dict)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_SCHEDULE, client_schedule)
        self.send
