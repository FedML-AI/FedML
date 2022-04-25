import logging
import os
import sys
import time

import json

from .FedEventSDK import FedEventSDK
from .FedLogsSDK import FedLogsSDK

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

from fedml_core.mlops_logger import MLOpsLogger
from .ClientStubObject import ClientStubObject
from .message_define import MyMessage

from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.server.server_manager import ServerManager
from fedml_core.distributed.communication.utils import log_round_start, log_round_end


class FedAVGServerManager(ServerManager):
    def __init__(
        self,
        args,
        dist_aggregator,
        comm=None,
        rank=0,
        size=0,
        backend="MPI",
        is_preprocessed=False,
        preprocessed_client_lists=None,
    ):
        super().__init__(args, comm, rank, size, backend)
        self.logs_sdk = None
        self.args = args
        self.dist_aggregator = dist_aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists
        self.client_stubs = {}
        self.client_online_mapping = dict()
        self.client_real_ids = json.loads(args.client_ids)

        self.mlops_logger = MLOpsLogger()
        self.mlops_logger.set_messenger(self.com_manager)
        self.dist_aggregator.aggregator.set_mlops_logger(self.mlops_logger)
        self.start_running_time = 0.0
        self.aggregated_model_url = None
        self.event_sdk = FedEventSDK(self.args)

    def run(self):
        super().run()

    def handle_messag_connection_ready(self, msg_params):
        logging.info("Connection is ready!")

    def send_init_msg(self):
        # sampling clients
        logging.info("send_init_msg")
        self.start_running_time = time.time()
        log_round_start(self.rank, 0)

        global_model_params = self.dist_aggregator.aggregator.get_global_model_params()

        client_id_list_in_this_round = self.dist_aggregator.aggregator.client_selection(
            self.round_idx, self.client_real_ids, self.args.client_num_per_round
        )
        data_silo_index_list = self.dist_aggregator.aggregator.data_silo_selection(
            self.round_idx, self.args.data_silo_num_in_total, len(client_id_list_in_this_round)
        )

        client_idx_in_this_round = 0
        for client_id in client_id_list_in_this_round:
            self.send_message_init_config(
                client_id, global_model_params, data_silo_index_list[client_idx_in_this_round]
            )
            client_idx_in_this_round += 1

        self.event_sdk.log_event_started("server.wait")

    def register_message_receive_handlers(self):
        print("register_message_receive_handlers------")
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_CONNECTION_IS_READY, self.handle_messag_connection_ready
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_CLIENT_STATUS, self.handle_message_client_status_update
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.handle_message_receive_model_from_client
        )

    def handle_message_client_status_update(self, msg_params):
        client_status = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_STATUS)
        client_os = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_OS)
        self.client_stubs[str(msg_params.get_sender_id())] = ClientStubObject(
            msg_params.get_sender_id(), client_os
        )
        if client_status == "ONLINE":
            self.client_online_mapping[str(msg_params.get_sender_id())] = True

        # notify MLOps with RUNNING status
        self.mlops_logger.report_server_training_status(self.args.run_id, MyMessage.MSG_MLOPS_SERVER_STATUS_RUNNING)

        all_client_is_online = True
        for client_id in self.client_real_ids:
            if not self.client_online_mapping.get(str(client_id), False):
                all_client_is_online = False
                break

        logging.info(
            "sender_id = %d, all_client_is_online = %s" % (msg_params.get_sender_id(), str(all_client_is_online))
        )

        if all_client_is_online:
            # send initialization message to all clients to start training
            self.send_init_msg()

    def handle_message_receive_model_from_client(self, msg_params):
        self.event_sdk.log_event_ended("comm_c2s", event_edge_id=0)
        self.event_sdk.log_event_ended("wait")

        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.dist_aggregator.aggregator.add_local_trained_result(
            self.client_real_ids.index(sender_id), model_params, local_sample_number
        )
        b_all_received = self.dist_aggregator.aggregator.check_whether_all_receive()
        logging.info("b_all_received = %s " % str(b_all_received))
        if b_all_received:
            self.event_sdk.log_event_started("aggregate")
            global_model_params = self.dist_aggregator.aggregator.aggregate()
            self.event_sdk.log_event_ended("aggregate")
            try:
                self.dist_aggregator.aggregator.test_on_server_for_all_clients(self.round_idx)
            except Exception as e:
                logging.info("aggregator.test exception: " + str(e) )

            # start the next round
            log_round_end(self.rank, self.round_idx)

            # send round info to the MQTT backend
            round_info = {"run_id": self.args.run_id,
                          "round_index": self.round_idx,
                          "total_rounds": self.round_num,
                          "running_time": round(time.time() - self.start_running_time, 4)}
            self.mlops_logger.report_server_training_round_info(round_info)

            client_id_list_in_this_round = self.dist_aggregator.aggregator.client_selection(
                self.round_idx, self.client_real_ids, self.args.client_num_per_round
            )
            data_silo_index_list = self.dist_aggregator.aggregator.data_silo_selection(
                self.round_idx, self.args.data_silo_num_in_total, len(client_id_list_in_this_round)
            )

            client_idx_in_this_round = 0
            for receiver_id in client_id_list_in_this_round:
                self.send_message_sync_model_to_client(
                    receiver_id, global_model_params, data_silo_index_list[client_idx_in_this_round]
                )
                client_idx_in_this_round += 1

            model_info = {
                "run_id": self.args.run_id,
                "round_idx": self.round_idx+1,
                "global_aggregated_model_s3_address": self.aggregated_model_url
            }
            self.mlops_logger.report_aggregated_model_info(model_info)
            self.aggregated_model_url = None

            self.round_idx += 1
            if self.round_idx == self.round_num:
                # post_complete_message_to_sweep_process(self.args)
                self.mlops_logger.report_server_id_status(self.args.run_id,
                                                          MyMessage.MSG_MLOPS_SERVER_STATUS_FINISHED)
                self.finish()
                self.dist_aggregator.cleanup_pg()
                return
            else:
                self.event_sdk.log_event_started("wait")

        log_round_start(self.rank, self.round_idx)

    def send_message_init_config(self, receive_id, global_model_params, client_index):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        client_stub_obj = self.client_stubs.get(str(receive_id), None)
        if client_stub_obj is not None:
            message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, client_stub_obj.os_name)
        else:
            message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "PythonClient")
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, data_silo_index):
        logging.info("send_message_sync_model_to_" "client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(data_silo_index))
        client_stub_obj = self.client_stubs.get(str(receive_id), None)
        if client_stub_obj is not None:
            message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, client_stub_obj.os_name)
        else:
            message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "PythonClient")
        self.send_message(message)

        if self.aggregated_model_url is None:
            self.aggregated_model_url = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL)
