import json
import time

from .message_define import MyMessage
from ...core.distributed.communication.message import Message
from ...core.distributed.server.server_manager import ServerManager
from ...core.mlops import MLOpsProfilerEvent, MLOpsMetrics
import logging

from ...core.mlops.mlops_configs import MLOpsConfigs


class FedMLServerManager(ServerManager):
    def __init__(
        self,
        args,
        aggregator_dist_adapter,
        comm=None,
        client_rank=0,
        client_num=0,
        backend="MQTT_S3",
        is_preprocessed=False,
        preprocessed_client_lists=None,
    ):
        super().__init__(args, comm, client_rank, client_num, backend)
        self.args = args
        self.aggregator_dist_adapter = aggregator_dist_adapter
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists

        self.pre_transform_model_file_path = args.global_model_file_path
        self.client_online_mapping = {}
        self.client_real_ids = json.loads(args.client_id_list)

        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_metrics = MLOpsMetrics()
            self.mlops_metrics.set_messenger(self.com_manager_status, args)
            self.mlops_event = MLOpsProfilerEvent(self.args)
            self.aggregator_dist_adapter.aggregator.set_mlops_logger(self.mlops_metrics)

        self.start_running_time = 0.0
        self.aggregated_model_url = None

    def run(self):
        # notify MLOps with RUNNING status
        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_metrics.report_server_training_status(
                self.args.run_id, MyMessage.MSG_MLOPS_SERVER_STATUS_RUNNING
            )

        super().run()

    def send_init_msg(self):
        # sampling clients
        self.start_running_time = time.time()

        global_model_params = (
            self.aggregator_dist_adapter.aggregator.get_global_model_params()
        )

        client_id_list_in_this_round = (
            self.aggregator_dist_adapter.aggregator.client_selection(
                self.round_idx, self.client_real_ids, self.args.client_num_per_round
            )
        )
        data_silo_index_list = (
            self.aggregator_dist_adapter.aggregator.data_silo_selection(
                self.round_idx,
                self.args.client_num_in_total,
                len(client_id_list_in_this_round),
            )
        )

        client_idx_in_this_round = 0
        for client_id in client_id_list_in_this_round:
            self.send_message_init_config(
                client_id,
                global_model_params,
                data_silo_index_list[client_idx_in_this_round],
            )
            client_idx_in_this_round += 1

        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_event.log_event_started("server.wait")

    def register_message_receive_handlers(self):
        print("register_message_receive_handlers------")
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_CONNECTION_IS_READY, self.handle_messag_connection_ready
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_CLIENT_STATUS,
            self.handle_message_client_status_update,
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.handle_message_receive_model_from_client,
        )

    def handle_messag_connection_ready(self, msg_params):
        logging.info("Connection is ready!")

    def handle_message_client_status_update(self, msg_params):
        client_status = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_STATUS)
        if client_status == "ONLINE":
            self.client_online_mapping[str(msg_params.get_sender_id())] = True

        # notify MLOps with RUNNING status
        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_metrics.report_server_training_status(
                self.args.run_id, MyMessage.MSG_MLOPS_SERVER_STATUS_RUNNING
            )

        all_client_is_online = True
        for client_id in self.client_real_ids:
            if not self.client_online_mapping.get(str(client_id), False):
                all_client_is_online = False
                break

        logging.info(
            "sender_id = %d, all_client_is_online = %s"
            % (msg_params.get_sender_id(), str(all_client_is_online))
        )

        if all_client_is_online:
            # send initialization message to all clients to start training
            self.send_init_msg()

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_event.log_event_ended("comm_c2s",  event_value=str(self.round_idx), event_edge_id=sender_id)

        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator_dist_adapter.aggregator.add_local_trained_result(
            self.client_real_ids.index(sender_id), model_params, local_sample_number
        )
        b_all_received = (
            self.aggregator_dist_adapter.aggregator.check_whether_all_receive()
        )
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            if hasattr(self.args, "backend") and self.args.using_mlops:
                self.mlops_event.log_event_started("aggregate")

            global_model_params = self.aggregator_dist_adapter.aggregator.aggregate()

            if hasattr(self.args, "backend") and self.args.using_mlops:
                self.mlops_event.log_event_ended("aggregate")
            try:
                self.aggregator_dist_adapter.aggregator.test_on_server_for_all_clients(
                    self.round_idx
                )
            except Exception as e:
                logging.info(
                    "aggregator_dist_adapter.aggregator.test exception: " + str(e)
                )

            # send round info to the MQTT backend
            if hasattr(self.args, "backend") and self.args.using_mlops:
                round_info = {
                    "run_id": self.args.run_id,
                    "round_index": self.round_idx,
                    "total_rounds": self.round_num,
                    "running_time": round(time.time() - self.start_running_time, 4),
                }
                self.mlops_metrics.report_server_training_round_info(round_info)

            client_id_list_in_this_round = (
                self.aggregator_dist_adapter.aggregator.client_selection(
                    self.round_idx, self.client_real_ids, self.args.client_num_per_round
                )
            )
            data_silo_index_list = (
                self.aggregator_dist_adapter.aggregator.data_silo_selection(
                    self.round_idx,
                    self.args.client_num_in_total,
                    len(client_id_list_in_this_round),
                )
            )

            client_idx_in_this_round = 0
            for receiver_id in client_id_list_in_this_round:
                self.send_message_sync_model_to_client(
                    receiver_id,
                    global_model_params,
                    data_silo_index_list[client_idx_in_this_round],
                )
                client_idx_in_this_round += 1

            if hasattr(self.args, "backend") and self.args.using_mlops:
                model_info = {
                    "run_id": self.args.run_id,
                    "round_idx": self.round_idx + 1,
                    "global_aggregated_model_s3_address": self.aggregated_model_url,
                }
                self.mlops_metrics.report_aggregated_model_info(model_info)
                self.aggregated_model_url = None

            self.round_idx += 1
            if self.round_idx == self.round_num:
                # post_complete_message_to_sweep_process(self.args)
                if hasattr(self.args, "backend") and self.args.using_mlops:
                    self.mlops_metrics.report_server_id_status(
                        self.args.run_id, MyMessage.MSG_MLOPS_SERVER_STATUS_FINISHED
                    )
                self.finish()
                return
            else:
                if hasattr(self.args, "backend") and self.args.using_mlops:
                    self.mlops_event.log_event_started("wait")

    def send_message_init_config(self, receive_id, global_model_params, datasilo_index):
        message = Message(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(datasilo_index))
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "PythonClient")
        self.send_message(message)

    def send_message_sync_model_to_client(
        self, receive_id, global_model_params, client_index
    ):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.get_sender_id(),
            receive_id,
        )
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, "PythonClient")
        self.send_message(message)

        if self.aggregated_model_url is None:
            # self.aggregated_model_url = message.get(
            #     MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL
            # )
            self.aggregated_model_url = "None"
