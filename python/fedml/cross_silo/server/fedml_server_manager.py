import json
import logging
import time

from fedml import mlops

from .message_define import MyMessage
from ...core import Context
from ...core.distributed.communication.message import Message
from ...core.distributed.fedml_comm_manager import FedMLCommManager
from ...core.mlops.mlops_profiler_event import MLOpsProfilerEvent


class FedMLServerManager(FedMLCommManager):
    def __init__(
            self, args, aggregator, comm=None, client_rank=0, client_num=0, backend="MQTT_S3",
    ):
        super().__init__(args, comm, client_rank, client_num, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.args.round_idx = 0

        self.client_online_mapping = {}
        self.client_real_ids = json.loads(args.client_id_list)

        self.is_initialized = False
        self.client_id_list_in_this_round = None
        self.data_silo_index_list = None

    def run(self):
        super().run()

    def send_init_msg(self):
        global_model_params = self.aggregator.get_global_model_params()

        global_model_url = None
        global_model_key = None

        client_idx_in_this_round = 0
        for client_id in self.client_id_list_in_this_round:
            global_model_url, global_model_key = self.send_message_init_config(
                client_id, global_model_params, self.data_silo_index_list[client_idx_in_this_round],
                global_model_url, global_model_key
            )
            client_idx_in_this_round += 1

        mlops.event("server.wait", event_started=True, event_value=str(self.args.round_idx))

        mlops.log_training_model_net_info(self.aggregator.aggregator.model)

    def register_message_receive_handlers(self):
        logging.info("register_message_receive_handlers------")
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_CONNECTION_IS_READY, self.handle_message_connection_ready
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_CLIENT_STATUS, self.handle_message_client_status_update,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.handle_message_receive_model_from_client,
        )

    def handle_message_connection_ready(self, msg_params):
        if not self.is_initialized:
            self.client_id_list_in_this_round = self.aggregator.client_selection(
                self.args.round_idx, self.client_real_ids, self.args.client_num_per_round
            )
            self.data_silo_index_list = self.aggregator.data_silo_selection(
                self.args.round_idx, self.args.client_num_in_total, len(self.client_id_list_in_this_round),
            )

            mlops.log_round_info(self.round_num, -1)

            # check client status in case that some clients start earlier than the server
            client_idx_in_this_round = 0
            for client_id in self.client_id_list_in_this_round:
                try:
                    self.send_message_check_client_status(
                        client_id, self.data_silo_index_list[client_idx_in_this_round],
                    )
                    logging.info("Connection ready for client" + str(client_id))
                except Exception as e:
                    logging.info("Connection not ready for client" + str(client_id))
                client_idx_in_this_round += 1

    def handle_message_client_status_update(self, msg_params):
        client_status = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_STATUS)
        if client_status == "ONLINE":
            self.client_online_mapping[str(msg_params.get_sender_id())] = True

            logging.info("self.client_online_mapping = {}".format(self.client_online_mapping))

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

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        mlops.event("comm_c2s", event_started=False, event_value=str(self.args.round_idx), event_edge_id=sender_id)

        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(
            self.client_real_ids.index(sender_id), model_params, local_sample_number
        )
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            mlops.event("server.wait", event_started=False, event_value=str(self.args.round_idx))
            mlops.event("server.agg_and_eval", event_started=True, event_value=str(self.args.round_idx))
            tick = time.time()
            global_model_params, model_list, model_list_idxes = self.aggregator.aggregate()

            logging.info("self.client_id_list_in_this_round = {}".format(self.client_id_list_in_this_round))
            new_client_id_list_in_this_round = []
            for client_idx in model_list_idxes:
                new_client_id_list_in_this_round.append(self.client_id_list_in_this_round[client_idx])
            logging.info("new_client_id_list_in_this_round = {}".format(new_client_id_list_in_this_round))
            Context().add(Context.KEY_CLIENT_ID_LIST_IN_THIS_ROUND, new_client_id_list_in_this_round)

            MLOpsProfilerEvent.log_to_wandb({"AggregationTime": time.time() - tick, "round": self.args.round_idx})

            self.aggregator.test_on_server_for_all_clients(self.args.round_idx)

            self.aggregator.assess_contribution()

            mlops.event("server.agg_and_eval", event_started=False, event_value=str(self.args.round_idx))

            # send round info to the MQTT backend
            mlops.log_round_info(self.round_num, self.args.round_idx)

            self.client_id_list_in_this_round = self.aggregator.client_selection(
                self.args.round_idx, self.client_real_ids, self.args.client_num_per_round
            )
            self.data_silo_index_list = self.aggregator.data_silo_selection(
                self.args.round_idx, self.args.client_num_in_total, len(self.client_id_list_in_this_round),
            )
            Context().add(Context.KEY_CLIENT_ID_LIST_IN_THIS_ROUND, self.client_id_list_in_this_round)

            if self.args.round_idx == 0:
                MLOpsProfilerEvent.log_to_wandb({"BenchmarkStart": time.time()})

            client_idx_in_this_round = 0
            global_model_url = None
            global_model_key = None
            for receiver_id in self.client_id_list_in_this_round:
                client_index = self.data_silo_index_list[client_idx_in_this_round]
                if type(global_model_params) is dict:
                    global_model_url, global_model_key = self.send_message_sync_model_to_client(
                        receiver_id, global_model_params[client_index], client_index, global_model_url,
                        global_model_key
                    )
                else:
                    global_model_url, global_model_key = self.send_message_sync_model_to_client(
                        receiver_id, global_model_params, client_index, global_model_url, global_model_key
                    )
                client_idx_in_this_round += 1

            self.args.round_idx += 1
            mlops.log_aggregated_model_info(
                self.args.round_idx, model_url=global_model_url,
                )

            logging.info("\n\n==========end {}-th round training===========\n".format(self.args.round_idx))
            mlops.event("server.wait", event_started=True, event_value=str(self.args.round_idx))

            if self.args.round_idx == self.round_num:
                logging.info("=============training is finished. Cleanup...============")
                self.cleanup()

    def cleanup(self):
        client_idx_in_this_round = 0
        for client_id in self.client_id_list_in_this_round:
            self.send_message_finish(
                client_id, self.data_silo_index_list[client_idx_in_this_round],
            )
            client_idx_in_this_round += 1
        time.sleep(3)
        self.finish()
        mlops.log_aggregation_finished_status()

    def send_message_init_config(self, receive_id, global_model_params, datasilo_index,
                                 global_model_url=None, global_model_key=None):
        tick = time.time()
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        if global_model_url is not None:
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL, global_model_url)
        if global_model_key is not None:
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_KEY, global_model_key)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "PythonClient")
        self.send_message(message)
        global_model_url = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL)
        global_model_key = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_KEY)
        MLOpsProfilerEvent.log_to_wandb({"Communiaction/Send_Total": time.time() - tick})
        return global_model_url, global_model_key

    def send_message_check_client_status(self, receive_id, datasilo_index):
        message = Message(MyMessage.MSG_TYPE_S2C_CHECK_CLIENT_STATUS, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
        self.send_message(message)

    def send_message_finish(self, receive_id, datasilo_index):
        message = Message(MyMessage.MSG_TYPE_S2C_FINISH, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
        self.send_message(message)
        logging.info(
            "finish from send id {} to receive id {}.".format(message.get_sender_id(), message.get_receiver_id()))
        logging.info(" ====================send cleanup message to {}====================".format(str(datasilo_index)))

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index,
                                          global_model_url=None, global_model_key=None):
        tick = time.time()
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id, )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        if global_model_url is not None:
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL, global_model_url)
        if global_model_key is not None:
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_KEY, global_model_key)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "PythonClient")
        self.send_message(message)

        MLOpsProfilerEvent.log_to_wandb({"Communiaction/Send_Total": time.time() - tick})

        global_model_url = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL)
        global_model_key = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_KEY)

        return global_model_url, global_model_key
