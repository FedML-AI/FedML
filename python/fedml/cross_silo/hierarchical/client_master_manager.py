# import logging
# import multiprocessing
# import os
# import platform
# import sys
# import time
# import uuid

# import json

# from ...model.mobile.model_transfer import mnn_pytorch

# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

# from fedml_core.mlops_logger import MLOpsLogger
# from .communication_manager import CommunicationManager
# from .message_define import MyMessage

# try:
#     from fedml_core.distributed.client.client_manager import ClientManager
#     from fedml_core.distributed.communication.message import Message
#     from fedml_core.distributed.communication.utils import log_round_start, log_round_end
# except ImportError:
#     from fedml_core.distributed.client.client_manager import ClientManager
#     from fedml_core.distributed.communication.message import Message
#     from fedml_core.distributed.communication.utils import log_round_start, log_round_end


from asyncio.log import logger
import json
import logging
import multiprocessing
import platform
import time

from .communication_manager import CommunicationManager
from .message_define import MyMessage

# from .utils import transform_list_to_tensor
from ...core.distributed.communication.message import Message
from ...core.mlops import MLOpsMetrics, MLOpsProfilerEvent
import torch.distributed as dist

from ...core.mlops.mlops_configs import MLOpsConfigs


class ClientMasterManager:
    def __init__(
        self, args, trainer_dist_adapter, comm=None, rank=0, size=0, backend="MPI"
    ):
        self.trainer_dist_adapter = trainer_dist_adapter
        self.args = args
        self.communication_manager = CommunicationManager(args, comm, rank, size, backend)
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.rank = rank
        self.client_real_ids = json.loads(args.client_id_list)
        # self.get_sender_id() is equal to the client rank (starting from 1)
        self.client_real_id = self.client_real_ids[
            self.communication_manager.get_sender_id() - 1
        ]

        self.has_sent_online_msg = False
        self.sys_stats_process = None

        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_metrics = MLOpsMetrics()
            self.mlops_metrics.set_messenger(self.com_manager_status, args)
            self.mlops_event = MLOpsProfilerEvent(self.args)

    def register_message_receive_handlers(self):
        self.communication_manager.register_message_receive_handler(
            MyMessage.MSG_TYPE_CONNECTION_IS_READY, self.handle_message_connection_ready
        )

        self.communication_manager.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init
        )
        self.communication_manager.register_message_receive_handler(
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
            MLOpsMetrics.report_sys_perf()

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        data_silo_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        logging.info("data_silo_index = %s" % str(data_silo_index))

        # Notify MLOps with training status.
        self.report_training_status(MyMessage.MSG_MLOPS_CLIENT_STATUS_TRAINING)

        self.sync_process_group(0, global_model_params, data_silo_index)

        self.trainer_dist_adapter.update_model(global_model_params)
        self.trainer_dist_adapter.update_dataset(int(data_silo_index))
        self.round_idx = 0

        # TODO: training to separate method
        logging.info("#######training########### round_id = %d" % self.round_idx)
        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_event.log_event_started("train")
        weights, local_sample_num = self.trainer_dist_adapter.train(self.round_idx)
        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_event.log_event_ended("train")
        self.send_model_to_server(0, weights, local_sample_num)

    def start_training(self):
        pass

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.sync_process_group(self.round_idx, model_params, client_index)

        self.trainer_dist_adapter.update_model(model_params)
        self.trainer_dist_adapter.update_dataset(int(client_index))

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
        logging.info("#######training########### round_id = %d" % self.round_idx)
        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_event.log_event_started("train")
        weights, local_sample_num = self.trainer_dist_adapter.train(self.round_idx)
        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_event.log_event_ended("train")
        self.send_model_to_server(0, weights, local_sample_num)

    def finish(self):
        logging.info(
            "Training finished for master client rank %s in silo %s"
            % (self.args.proc_rank_in_silo, self.args.rank_in_node)
        )

        self.trainer_dist_adapter.cleanup_pg()

        # Notify MLOps with the finished message
        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_metrics.report_client_id_status(
                self.args.run_id,
                self.client_real_id,
                MyMessage.MSG_MLOPS_CLIENT_STATUS_FINISHED,
            )

        self.communication_manager.finish()

        mlops_metrics = MLOpsMetrics()
        mlops_metrics.set_sys_reporting_status(False)
    # def exit_program(self):
    #     try:
    #         sys.exit(100)
    #     except SystemExit as e:
    #         exit(100)

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        if hasattr(self.args, "backend") and self.args.using_mlops:
            self.mlops_event.log_event_started("comm_c2s", event_value=str(self.round_idx), )
        message = Message(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
            self.client_real_id,
            receive_id,
        )

        model_url = "None"

        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL, model_url)
        self.communication_manager.send_message(message)

        # Report client model to MLOps
        if hasattr(self.args, "backend") and self.args.using_mlops:
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
        self.communication_manager.send_message(message)

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

    def sync_process_group(
        self, round_idx, model_params=None, client_index=None, src=0
    ):
        logging.info("sending round number to pg")
        round_number = [round_idx, model_params, client_index]
        dist.broadcast_object_list(
            round_number,
            src=src,
            group=self.trainer_dist_adapter.process_group_manager.get_process_group(),
        )
        logging.info("round number %d broadcasted to process group" % round_number[0])

    def run(self):
        self.register_message_receive_handlers()
        self.communication_manager.run()

        ###############################33

        logging.info("Connection is ready!")
        if not self.has_sent_online_msg:
            self.has_sent_online_msg = True
            self.send_client_status(0)

            # Notify MLOps with training status.
            self.report_training_status(MyMessage.MSG_MLOPS_CLIENT_STATUS_INITIALIZING)
