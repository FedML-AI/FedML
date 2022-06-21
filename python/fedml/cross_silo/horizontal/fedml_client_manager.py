import json
import logging
import multiprocessing
import platform
import time

from .message_define import MyMessage
from .utils import transform_list_to_tensor
from ...core.distributed.client.client_manager import ClientManager
from ...core.distributed.communication.message import Message
from ...core.mlops import MLOpsMetrics, MLOpsProfilerEvent


class FedMLClientManager(ClientManager):
    def __init__(
        self, args, trainer, comm=None, client_rank=0, client_num=0, backend="MPI"
    ):
        super().__init__(args, comm, client_rank, client_num, backend)
        self.args = args
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

        self.client_real_ids = json.loads(args.client_id_list)
        logging.info("self.client_real_ids = {}".format(self.client_real_ids))
        # for the client, len(self.client_real_ids)==1: we only specify its client id in the list, not including others.
        self.client_real_id = self.client_real_ids[0]

        self.has_sent_online_msg = False
        self.sys_stats_process = None

        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_metrics = MLOpsMetrics()
            self.mlops_metrics.set_messenger(self.com_manager_status, args)
            self.mlops_event = MLOpsProfilerEvent(self.args)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_CONNECTION_IS_READY, self.handle_message_connection_ready
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_CHECK_CLIENT_STATUS, self.handle_message_check_status
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init
        )

        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.handle_message_receive_model_from_server,
        )

    def handle_message_connection_ready(self, msg_params):
        logging.info("Connection is ready!")
        if not self.has_sent_online_msg:
            self.has_sent_online_msg = True
            self.send_client_status(0)

            # Notify MLOps with training status.
            self.report_training_status(MyMessage.MSG_MLOPS_CLIENT_STATUS_INITIALIZING)

            # Open new process for report system performances to MQTT server
            self.sys_stats_process = multiprocessing.Process(
                target=self.report_sys_performances
            )
            self.sys_stats_process.start()

    def handle_message_check_status(self, msg_params):
        self.send_client_status(0)

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        global_model_params = transform_list_to_tensor(global_model_params)

        logging.info("client_index = %s" % str(client_index))

        # Notify MLOps with training status.
        self.report_training_status(MyMessage.MSG_MLOPS_CLIENT_STATUS_TRAINING)

        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index))
        self.round_idx = 0
        self.__train()

    def start_training(self):
        self.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        model_params = transform_list_to_tensor(model_params)

        self.trainer.update_model(model_params)
        self.trainer.update_dataset(int(client_index))

        if self.round_idx == self.num_rounds - 1:

            # Notify MLOps with the finished message
            if hasattr(self.args, "backend") and self.args.using_mlops:
                self.mlops_metrics.report_client_id_status(
                    self.args.run_id,
                    self.client_real_id,
                    MyMessage.MSG_MLOPS_CLIENT_STATUS_FINISHED,
                )

            self.finish()
            return
        self.round_idx += 1
        self.__train()

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_event.log_event_started(
                "comm_c2s", event_value=str(self.round_idx)
            )
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.client_real_id,
            receive_id,
        )

        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

        # Report client model to MLOps
        if hasattr(self.args, "backend") and self.args.using_mlops:
            model_url = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL)
            model_info = {
                "run_id": self.args.run_id,
                "edge_id": self.client_real_id,
                "round_idx": self.round_idx + 1,
                "client_model_s3_address": model_url,
            }
            self.mlops_metrics.report_client_model_info(model_info)

    def send_client_status(self, receive_id, status="ONLINE"):
        logging.info("send_client_status")
        message = Message(
            MyMessage.MSG_TYPE_C2S_CLIENT_STATUS, self.client_real_id, receive_id
        )
        sys_name = platform.system()
        if sys_name == "Darwin":
            sys_name = "Mac"
        # Debug for simulation mobile system
        # sys_name = MyMessage.MSG_CLIENT_OS_ANDROID

        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_STATUS, status)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, sys_name)
        self.send_message(message)

    def report_training_status(self, status):
        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_metrics.report_client_training_status(
                self.client_real_id, status
            )

    def report_sys_performances(self):
        if hasattr(self.args, "backend") and self.args.using_mlops:
            while self.round_idx != self.num_rounds - 1:
                # Notify MLOps with system information.
                self.mlops_metrics.report_system_metric()
                time.sleep(30)

    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_event.log_event_started("train", event_value=str(self.round_idx))

        time.sleep(10)
        weights, local_sample_num = self.trainer.train(self.round_idx)

        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_event.log_event_ended("train", event_value=str(self.round_idx))

        self.send_model_to_server(0, weights, local_sample_num)

    def run(self):
        super().run()
