import json
import logging
import time
from time import sleep

import numpy as np

from fedml import mlops
from .sa_message_define import MyMessage
from ...core.distributed.fedml_comm_manager import FedMLCommManager
from ...core.distributed.communication.message import Message


class FedMLServerManager(FedMLCommManager):
    def __init__(
        self,
        args,
        aggregator,
        comm=None,
        client_rank=0,
        client_num=0,
        backend="MQTT_S3",
        is_preprocessed=False,
        preprocessed_client_lists=None,
    ):
        """
        Initialize the Federated Learning Server Manager.

        Args:
            args (object): Arguments object containing configuration parameters.
            aggregator (object): Federated learning aggregator.
            comm (object, optional): Communication manager (default: None).
            client_rank (int, optional): Rank of the client (default: 0).
            client_num (int, optional): Number of clients (default: 0).
            backend (str, optional): Backend for communication (default: "MQTT_S3").
            is_preprocessed (bool, optional): Whether the data is preprocessed (default: False).
            preprocessed_client_lists (list, optional): List of preprocessed clients (default: None).
        """
        super().__init__(args, comm, client_rank, client_num, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists

        self.client_online_mapping = {}
        self.client_real_ids = json.loads(args.client_id_list)

        self.active_clients_first_round = []
        self.second_round_clients = []

        ### new added parameters in main file ###
        self.targeted_number_active_clients = args.worker_num
        self.privacy_guarantee = int(np.floor(args.worker_num / 2))
        self.prime_number = args.prime_number
        self.precision_parameter = args.precision_parameter
        self.public_keys_received = 0
        self.ss_received = 0
        self.num_pk_per_user = 2
        self.public_key_list = np.empty(
            shape=(self.num_pk_per_user,
                   self.targeted_number_active_clients), dtype="int64"
        )
        self.b_u_SS_list = np.empty(
            (self.targeted_number_active_clients,
             self.targeted_number_active_clients), dtype="int64"
        )
        self.s_sk_SS_list = np.empty(
            (self.targeted_number_active_clients,
             self.targeted_number_active_clients), dtype="int64"
        )
        self.SS_rx = np.empty((self.targeted_number_active_clients,
                              self.targeted_number_active_clients), dtype="int64")

        self.aggregated_model_url = None

        self.is_initialized = False
        self.client_id_list_in_this_round = None
        self.data_silo_index_list = None

    def run(self):
        """
        Start the Federated Learning Server Manager.

        This method starts the server manager and begins the federated learning process.
        """
        super().run()

    def send_init_msg(self):
        """
        Send initialization messages to clients.

        This method sends initialization messages to all clients, providing them with the
        global model parameters to start training.

        Args:
            None

        Returns:
            None
        """
        global_model_params = self.aggregator.get_global_model_params()

        client_idx_in_this_round = 0
        for client_id in self.client_id_list_in_this_round:
            self.send_message_init_config(
                client_id, global_model_params, self.data_silo_index_list[client_idx_in_this_round],
            )
            client_idx_in_this_round += 1

        mlops.event("server.wait", event_started=True,
                    event_value=str(self.round_idx))

    def register_message_receive_handlers(self):
        """
        Register message receive handlers for server communication.

        This method registers various message receive handlers for different types of
        communication messages received by the server.

        Args:
            None

        Returns:
            None
        """
        print("register_message_receive_handlers------")
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_CONNECTION_IS_READY, self.handle_messag_connection_ready
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_CLIENT_STATUS, self.handle_message_client_status_update,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.handle_message_receive_model_from_client,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_PK_TO_SERVER, self._handle_message_receive_public_key,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_SS_TO_SERVER, self._handle_message_receive_ss,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_SS_OTHERS_TO_SERVER, self._handle_message_receive_ss_others_from_client,
        )

    def handle_messag_connection_ready(self, msg_params):
        """
        Handle a connection-ready message from clients.

        This function processes client connection requests and initializes necessary
        parameters for the server's operation.

        Args:
            self: The server instance.
            msg_params (dict): A dictionary containing message parameters.

        Returns:
            None
        """
        self.client_id_list_in_this_round = self.aggregator.client_selection(
            self.round_idx, self.client_real_ids, self.args.client_num_per_round
        )
        self.data_silo_index_list = self.aggregator.data_silo_selection(
            self.round_idx, self.args.client_num_in_total, len(
                self.client_id_list_in_this_round),
        )
        if not self.is_initialized:
            mlops.log_round_info(self.round_num, -1)

            # check client status in case that some clients start earlier than the server
            client_idx_in_this_round = 0
            for client_id in self.client_id_list_in_this_round:
                self.send_message_check_client_status(
                    client_id, self.data_silo_index_list[client_idx_in_this_round],
                )
                client_idx_in_this_round += 1

    def handle_message_client_status_update(self, msg_params):
        """
        Handle a message containing client status updates.

        This function updates the server's record of client statuses and takes
        appropriate actions when all clients are online.

        Args:
            self: The server instance.
            msg_params (dict): A dictionary containing message parameters.

        Returns:
            None
        """
        client_status = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_STATUS)
        if client_status == "ONLINE":
            self.client_online_mapping[str(msg_params.get_sender_id())] = True

        mlops.log_aggregation_status(MyMessage.MSG_MLOPS_SERVER_STATUS_RUNNING)

        all_client_is_online = True
        for client_id in self.client_id_list_in_this_round:
            if not self.client_online_mapping.get(str(client_id), False):
                all_client_is_online = False
                break

        logging.info(
            "sender_id = %d, all_client_is_online = %s" % (
                msg_params.get_sender_id(), str(all_client_is_online))
        )

        if all_client_is_online:
            # send initialization message to all clients to start training
            self.send_init_msg()
            self.is_initialized = True

    def _handle_message_receive_public_key(self, msg_params):
        """
        Handle the reception of public keys from clients.

        This function receives and processes public keys from active clients,
        combining them for further use.

        Args:
            self: The server instance.
            msg_params (dict): A dictionary containing message parameters.

        Returns:
            None
        """
        # Receive the aggregate of encoded masks for active clients
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        public_key = msg_params.get(MyMessage.MSG_ARG_KEY_PK)
        self.public_key_list[:, sender_id - 1] = public_key
        self.public_keys_received += 1
        if self.public_keys_received == self.targeted_number_active_clients:
            data = np.reshape(
                self.public_key_list, self.num_pk_per_user * self.targeted_number_active_clients)
            for i in range(self.targeted_number_active_clients):
                logging.info("sending data = {}".format(data))
                self._send_public_key_others_to_user(i + 1, data)

    def _handle_message_receive_ss(self, msg_params):
        """
        Handle the reception of encoded masks from clients.

        This function receives and processes encoded masks from active clients,
        aggregating them for further use.

        Args:
            self: The server instance.
            msg_params (dict): A dictionary containing message parameters.

        Returns:
            None
        """
        # Receive the aggregate of encoded masks for active clients
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        b_u_SS = msg_params.get(MyMessage.MSG_ARG_KEY_B_SS)
        s_sk_SS = msg_params.get(MyMessage.MSG_ARG_KEY_SK_SS)
        self.b_u_SS_list[sender_id - 1, :] = b_u_SS
        self.s_sk_SS_list[sender_id - 1, :] = s_sk_SS
        self.ss_received += 1
        if self.ss_received == self.targeted_number_active_clients:
            for i in range(self.targeted_number_active_clients):
                self._send_ss_others_to_user(
                    i + 1, self.b_u_SS_list[:, i], self.s_sk_SS_list[:, i])

    def handle_message_receive_model_from_client(self, msg_params):
        """
        Handle the reception of a trained model from a client.

        This function receives and processes a trained model from a client,
        updating the server's records and taking appropriate actions.

        Args:
            self: The server instance.
            msg_params (dict): A dictionary containing message parameters.

        Returns:
            None
        """
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        mlops.event(
            "comm_c2s", event_started=False, event_value=str(self.round_idx), event_edge_id=sender_id,
        )

        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(
            self.client_real_ids.index(
                sender_id), model_params, local_sample_number
        )
        self.active_clients_first_round.append(sender_id - 1)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        # TODO: add a timeout procedure here
        if b_all_received:
            # Specify the active clients for the first round and inform them
            for receiver_id in range(1, self.size + 1):
                self._send_message_to_active_client(
                    receiver_id, self.active_clients_first_round)

    def _handle_message_receive_ss_others_from_client(self, msg_params):
        """
        Handle the reception of encoded masks from clients in the second round.

        This function receives and processes encoded masks from clients in the
        second round, and performs model aggregation and evaluation.

        Args:
            self: The server instance.
            msg_params (dict): A dictionary containing message parameters.

        Returns:
            None
        """
        # Receive the aggregate of encoded masks for active clients
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        ss_others = msg_params.get(MyMessage.MSG_ARG_KEY_SS_OTHERS)

        self.SS_rx[:, sender_id - 1] = ss_others
        self.second_round_clients.append(sender_id - 1)
        # logging.info("Server: mask_all_received = " + str(b_all_received) + " in round_idx %d" % self.round_idx)

        # After receiving enough aggregate of encoded masks, server recovers the aggregate-model
        if len(self.second_round_clients) == len(self.active_clients_first_round):
            # Secure Model Aggregation
            global_model_params = self.aggregator.aggregate_model_reconstruction(
                self.active_clients_first_round, self.second_round_clients, self.SS_rx, self.public_key_list
            )
            # evaluation
            try:
                self.aggregator.test_on_server_for_all_clients(self.round_idx)
            except Exception as e:
                logging.info("aggregator.test exception: " + str(e))

            self.client_id_list_in_this_round = self.aggregator.client_selection(
                self.round_idx, self.client_real_ids, self.args.client_num_per_round
            )
            self.data_silo_index_list = self.aggregator.data_silo_selection(
                self.round_idx, self.args.client_num_in_total, len(
                    self.client_id_list_in_this_round),
            )

            client_idx_in_this_round = 0
            for receiver_id in self.client_id_list_in_this_round:
                self.send_message_sync_model_to_client(
                    receiver_id, global_model_params, self.data_silo_index_list[
                        client_idx_in_this_round],
                )
                client_idx_in_this_round += 1

            # start the next round
            self.aggregated_model_url = None
            self.round_idx += 1
            self.active_clients_first_round = []
            self.second_round_clients = []
            self.public_keys_received = 0
            self.ss_received = 0
            self.num_pk_per_user = 2
            self.public_key_list = np.empty(
                shape=(self.num_pk_per_user,
                       self.targeted_number_active_clients), dtype="int64"
            )
            self.b_u_SS_list = np.empty(
                (self.targeted_number_active_clients,
                 self.targeted_number_active_clients), dtype="int64"
            )
            self.s_sk_SS_list = np.empty(
                (self.targeted_number_active_clients,
                 self.targeted_number_active_clients), dtype="int64"
            )
            self.SS_rx = np.empty(
                (self.targeted_number_active_clients,
                 self.targeted_number_active_clients), dtype="int64"
            )

            if self.round_idx == self.round_num:
                logging.info(
                    "=================TRAINING IS FINISHED!=============")
                sleep(3)
                self.finish()
            if self.is_preprocessed:
                mlops.log_training_finished_status()
                logging.info(
                    "=============training is finished. Cleanup...============")
                self.cleanup()
            else:
                logging.info("waiting for another round...")
                mlops.event("server.wait", event_started=True,
                            event_value=str(self.round_idx))

    def cleanup(self):
        """
        Cleanup function to finish the training process.

        This function is responsible for cleaning up after the training process,
        sending finish messages to clients, and finalizing the server's state.

        Args:
            self: The server instance.

        Returns:
            None
        """

        client_idx_in_this_round = 0
        for client_id in self.client_id_list_in_this_round:
            self.send_message_finish(
                client_id, self.data_silo_index_list[client_idx_in_this_round],
            )
            client_idx_in_this_round += 1
        time.sleep(3)
        self.finish()

    def send_message_init_config(self, receive_id, global_model_params, datasilo_index):
        """
        Send an initialization configuration message to a client.

        This function sends an initialization message containing global model
        parameters and other configuration details to a specific client.

        Args:
            self: The server instance.
            receive_id (int): The ID of the receiving client.
            global_model_params (dict): Global model parameters.
            datasilo_index (int): The index of the data silo associated with the client.

        Returns:
            None
        """
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                          self.get_sender_id(), receive_id)
        message.add_params(
            MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(
            MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "PythonClient")
        self.send_message(message)

    def send_message_check_client_status(self, receive_id, datasilo_index):
        """
        Send a message to check the status of a client.

        This function sends a message to a client to check its status and readiness.

        Args:
            self: The server instance.
            receive_id (int): The ID of the receiving client.
            datasilo_index (int): The index of the data silo associated with the client.

        Returns:
            None
        """

        message = Message(MyMessage.MSG_TYPE_S2C_CHECK_CLIENT_STATUS,
                          self.get_sender_id(), receive_id)
        message.add_params(
            MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
        self.send_message(message)

    def send_message_finish(self, receive_id, datasilo_index):
        """
        Send a finish message to a client.

        This function sends a finish message to a client to signal the end of the
        training process.

        Args:
            self: The server instance.
            receive_id (int): The ID of the receiving client.
            datasilo_index (int): The index of the data silo associated with the client.

        Returns:
            None
        """
        message = Message(MyMessage.MSG_TYPE_S2C_FINISH,
                          self.get_sender_id(), receive_id)
        message.add_params(
            MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
        self.send_message(message)
        logging.info(" ====================send cleanup message to {}====================".format(
            str(datasilo_index)))

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index):
        """
        Send a message to synchronize the global model with a client.

        This function sends a synchronization message to a specific client,
        containing the global model parameters and client index.

        Args:
            self: The server instance.
            receive_id (int): The ID of the receiving client.
            global_model_params (dict): Global model parameters.
            client_index (int): The index of the client.

        Returns:
            None
        """
        logging.info(
            "send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id,)
        message.add_params(
            MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(
            MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "PythonClient")
        self.send_message(message)

        mlops.log_aggregated_model_info(
            self.round_idx + 1, model_url=message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL),
        )

    def _send_public_key_others_to_user(self, receive_id, public_key_other):
        """
        Send public keys to a user/client.

        This function sends public keys to a specific user/client, typically during
        a secure communication setup.

        Args:
            self: The server instance.
            receive_id (int): The ID of the receiving user/client.
            public_key_other: The public keys to send.

        Returns:
            None
        """

        logging.info(
            "Server send_message_to_active_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_OTHER_PK_TO_CLIENT,
                          self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_PK_OTHERS, public_key_other)
        self.send_message(message)

    def _send_ss_others_to_user(self, receive_id, b_ss_others, sk_ss_others):
        """
        Send encoded masks to a user/client.

        This function sends encoded masks to a specific user/client, typically during
        a secure communication setup.

        Args:
            self: The server instance.
            receive_id (int): The ID of the receiving user/client.
            b_ss_others: Encoded masks (b values) to send.
            sk_ss_others: Encoded masks (sk values) to send.

        Returns:
            None
        """
        logging.info(
            "Server send_message_to_active_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_OTHER_SS_TO_CLIENT,
                          self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_B_SS_OTHERS, b_ss_others)
        message.add_params(MyMessage.MSG_ARG_KEY_SK_SS_OTHERS, sk_ss_others)
        self.send_message(message)

    def _send_message_to_active_client(self, receive_id, active_clients):
        """
        Send a message to active clients.

        This function sends a message to a specific user/client containing a list of
        active clients, typically during initialization.

        Args:
            self: The server instance.
            receive_id (int): The ID of the receiving user/client.
            active_clients (list): A list of active client IDs.

        Returns:
            None
        """
        logging.info(
            "Server send_message_to_active_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_ACTIVE_CLIENT_LIST,
                          self.get_sender_id(), receive_id)
        message.add_params(
            MyMessage.MSG_ARG_KEY_ACTIVE_CLIENTS, active_clients)
        self.send_message(message)
