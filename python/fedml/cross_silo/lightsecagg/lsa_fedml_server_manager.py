import json
import logging
import time
from time import sleep

import numpy as np

from fedml import mlops
from .lsa_message_define import MyMessage
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
        self.active_clients_second_round = []

        ### new added parameters in main file ###
        # self.targeted_number_active_clients = args.targeted_number_active_clients
        # self.privacy_guarantee = args.privacy_guarantee
        self.targeted_number_active_clients = args.worker_num
        self.privacy_guarantee = int(np.floor(args.worker_num / 2))
        self.prime_number = args.prime_number
        self.precision_parameter = args.precision_parameter

        self.aggregated_model_url = None

        self.is_initialized = False
        self.client_id_list_in_this_round = None
        self.data_silo_index_list = None

    def run(self):
        super().run()

    def send_init_msg(self):
        global_model_params = self.aggregator.get_global_model_params()

        client_idx_in_this_round = 0
        for client_id in self.client_id_list_in_this_round:
            self.send_message_init_config(
                client_id, global_model_params, self.data_silo_index_list[client_idx_in_this_round],
            )
            client_idx_in_this_round += 1

        mlops.event("server.wait", event_started=True, event_value=str(self.round_idx))

    def register_message_receive_handlers(self):
        print("register_message_receive_handlers------")
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_CONNECTION_IS_READY, self.handle_messag_connection_ready
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_CLIENT_STATUS, self.handle_message_client_status_update,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_ENCODED_MASK_TO_SERVER, self.handle_message_receive_encoded_mask_from_client,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.handle_message_receive_model_from_client,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MASK_TO_SERVER, self.handle_message_receive_aggregate_encoded_mask_from_client,
        )

    def handle_messag_connection_ready(self, msg_params):
        self.client_id_list_in_this_round = self.aggregator.client_selection(
            self.round_idx, self.client_real_ids, self.args.client_num_per_round
        )
        self.data_silo_index_list = self.aggregator.data_silo_selection(
            self.round_idx, self.args.client_num_in_total, len(self.client_id_list_in_this_round),
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
            "sender_id = %d, all_client_is_online = %s" % (msg_params.get_sender_id(), str(all_client_is_online))
        )

        if all_client_is_online:
            # send initialization message to all clients to start training
            self.send_init_msg()
            self.is_initialized = True

    def handle_message_receive_encoded_mask_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        receive_id = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_ID)
        encoded_mask = msg_params.get(MyMessage.MSG_ARG_KEY_ENCODED_MASK)
        self.send_message_encoded_mask_to_client(sender_id, receive_id, encoded_mask)

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        mlops.event(
            "comm_c2s", event_started=False, event_value=str(self.round_idx), event_edge_id=sender_id,
        )

        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(
            self.client_real_ids.index(sender_id), model_params, local_sample_number
        )
        self.active_clients_first_round.append(sender_id - 1)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        # TODO: add a timeout procedure here
        if b_all_received:
            # Specify the active clients for the first round and inform them
            for receiver_id in range(1, self.size + 1):
                self.send_message_to_active_client(receiver_id, self.active_clients_first_round)

    def handle_message_receive_aggregate_encoded_mask_from_client(self, msg_params):
        # Receive the aggregate of encoded masks for active clients
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        aggregate_encoded_mask = msg_params.get(MyMessage.MSG_ARG_KEY_AGGREGATE_ENCODED_MASK)
        self.aggregator.add_local_aggregate_encoded_mask(sender_id - 1, aggregate_encoded_mask)
        logging.info(
            "Server handle_message_receive_aggregate_mask = %d from_client =  %d"
            % (len(aggregate_encoded_mask), sender_id)
        )
        # Active clients for the second round
        self.active_clients_second_round.append(sender_id - 1)
        b_all_received = self.aggregator.check_whether_all_aggregate_encoded_mask_receive()
        logging.info("Server: mask_all_received = " + str(b_all_received) + " in round_idx %d" % self.round_idx)

        # TODO: add a timeout step
        # After receiving enough aggregate of encoded masks, server recovers the aggregate-model
        if b_all_received:
            mlops.event("server.wait", event_started=False, event_value=str(self.round_idx))
            mlops.event(
                "server.agg_and_eval", event_started=True, event_value=str(self.round_idx),
            )

            # Secure Model Aggregation
            global_model_params = self.aggregator.aggregate_model_reconstruction(
                self.active_clients_first_round, self.active_clients_second_round
            )
            # evaluation
            try:
                self.aggregator.test_on_server_for_all_clients(self.round_idx)
            except Exception as e:
                logging.info("aggregator.test exception: " + str(e))

            mlops.event(
                "server.agg_and_eval", event_started=False, event_value=str(self.round_idx),
            )

            mlops.log_round_info(self.round_num, self.round_idx)

            self.client_id_list_in_this_round = self.aggregator.client_selection(
                self.round_idx, self.client_real_ids, self.args.client_num_per_round
            )
            self.data_silo_index_list = self.aggregator.data_silo_selection(
                self.round_idx, self.args.client_num_in_total, len(self.client_id_list_in_this_round),
            )

            client_idx_in_this_round = 0
            for receiver_id in self.client_id_list_in_this_round:
                self.send_message_sync_model_to_client(
                    receiver_id, global_model_params, self.data_silo_index_list[client_idx_in_this_round],
                )
                client_idx_in_this_round += 1

            mlops.log_aggregated_model_info(self.round_idx + 1, self.aggregated_model_url)
            self.aggregated_model_url = None

            # start the next round
            self.round_idx += 1
            self.active_clients_first_round = []
            self.active_clients_second_round = []

            if self.round_idx == self.round_num:
                logging.info("=================TRAINING IS FINISHED!=============")
                sleep(3)
                self.finish()
            if self.is_preprocessed:
                mlops.log_training_finished_status()
                logging.info("=============training is finished. Cleanup...============")
                self.cleanup()
            else:
                logging.info("waiting for another round...")
                mlops.event("server.wait", event_started=True, event_value=str(self.round_idx))

    def cleanup(self):

        client_idx_in_this_round = 0
        for client_id in self.client_id_list_in_this_round:
            self.send_message_finish(
                client_id, self.data_silo_index_list[client_idx_in_this_round],
            )
            client_idx_in_this_round += 1
        time.sleep(3)
        self.finish()

    def send_message_init_config(self, receive_id, global_model_params, datasilo_index):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "PythonClient")
        self.send_message(message)

    def send_message_encoded_mask_to_client(self, sender_id, receive_id, encoded_mask):
        message = Message(MyMessage.MSG_TYPE_S2C_ENCODED_MASK_TO_CLIENT, self.get_sender_id(), receive_id,)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_ID, sender_id)
        message.add_params(MyMessage.MSG_ARG_KEY_ENCODED_MASK, encoded_mask)
        self.send_message(message)

    def send_message_check_client_status(self, receive_id, datasilo_index):
        message = Message(MyMessage.MSG_TYPE_S2C_CHECK_CLIENT_STATUS, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
        self.send_message(message)

    def send_message_finish(self, receive_id, datasilo_index):
        message = Message(MyMessage.MSG_TYPE_S2C_FINISH, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
        self.send_message(message)
        logging.info(" ====================send cleanup message to {}====================".format(str(datasilo_index)))

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id,)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "PythonClient")
        self.send_message(message)

        mlops.log_aggregated_model_info(
            self.round_idx + 1, model_url=message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL),
        )

    def send_message_to_active_client(self, receive_id, active_clients):
        logging.info("Server send_message_to_active_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SEND_TO_ACTIVE_CLIENT, self.get_sender_id(), receive_id,)
        message.add_params(MyMessage.MSG_ARG_KEY_ACTIVE_CLIENTS, active_clients)
        self.send_message(message)
