import logging
import multiprocessing
import os
import platform
import sys
import time

import json
import trace
import traceback

from .FedEventSDK import FedEventSDK
from .FedLogsSDK import FedLogsSDK

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

from fedml_core.mlops_logger import MLOpsLogger
from .CommunicationManager import CommunicationManager
from .message_define import MyMessage

try:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
    from fedml_core.distributed.communication.utils import log_round_start, log_round_end
except ImportError:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
    from fedml_core.distributed.communication.utils import log_round_start, log_round_end


class ClientMasterManager:
    def __init__(self, args, dist_worker, comm=None, rank=0, size=0, backend="MPI"):
        self.event_sdk = None
        self.dist_worker = dist_worker
        logging.info("ClientMasterManager args client_ids: " + args.client_ids)
        self.communication_manager = CommunicationManager(args, comm, rank, size, backend)
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.args = args
        self.rank = rank
        self.client_real_ids = json.loads(args.client_ids)
        if self.communication_manager.get_sender_id() <= len(self.client_real_ids):
            self.client_real_id = self.client_real_ids[self.communication_manager.get_sender_id() - 1]
        else:
            self.client_real_id = self.client_real_ids[0]

        self.has_sent_online_msg = False
        self.sys_stats_process = None
        self.mlops_logger = MLOpsLogger()
        self.mlops_logger.set_messenger(self.communication_manager.com_manager_status, args)
        self.event_sdk = FedEventSDK(self.args)

    def register_message_receive_handlers(self):
        self.communication_manager.register_message_receive_handler(
            MyMessage.MSG_TYPE_CONNECTION_IS_READY, self.handle_messag_connection_ready
        )
        self.communication_manager.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init
        )
        self.communication_manager.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.handle_message_receive_model_from_server
        )

    def handle_messag_connection_ready(self, msg_params):
        logging.info("Connection is ready!")
        if not self.has_sent_online_msg:
            self.has_sent_online_msg = True
            self.send_client_status(0)

            # Notify MLOps with training status.
            self.report_training_status(MyMessage.MSG_MLOPS_CLIENT_STATUS_INITIALIZING)

            # Open new process for report system performances to MQTT server
            self.sys_stats_process = multiprocessing.Process(target=self.report_sys_performances)
            self.sys_stats_process.start()

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        data_silo_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        logging.info("data_silo_index = %s" % str(data_silo_index))



        # Notify MLOps with training status.
        self.report_training_status(MyMessage.MSG_MLOPS_CLIENT_STATUS_TRAINING)

        if self.args.is_mobile != 1:
            self.dist_worker.update_model(global_model_params)
            self.dist_worker.update_dataset(int(data_silo_index))
        self.round_idx = 0
        if self.args.is_mobile != 1:
            weights, local_sample_num = self.dist_worker.train(self.round_idx)
            self.send_model_to_server(0, weights, local_sample_num)
        else:
            self.send_model_to_server(0, global_model_params, 100)

    def start_training(self):
        pass

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)


        if self.args.is_mobile != 1:
            self.dist_worker.update_model(model_params)
            self.dist_worker.update_dataset(int(client_index))

        if self.round_idx == self.num_rounds - 1:
            logging.info("+++++++++++++++++++++++++")
            self.finish()
            return
        else:
            logging.info("-----------------------")

        self.round_idx += 1
        if self.args.is_mobile != 1:
            self.event_sdk.log_event_started("train")
            weights, local_sample_num = self.dist_worker.train(self.round_idx)
            self.event_sdk.log_event_ended("train")
            self.send_model_to_server(0, weights, local_sample_num)
        else:
            self.send_model_to_server(0, model_params, 100)

    def finish(self):
        logging.info(
            "Training finished for master client rank %s in silo %s" % (self.args.silo_proc_rank, self.args.silo_rank)
        )

        self.dist_worker.cleanup_pg()

        # Notify MLOps with the finished message
        self.mlops_logger.report_client_id_status(self.args.run_id, self.client_real_id,
                                                  MyMessage.MSG_MLOPS_CLIENT_STATUS_FINISHED)

        self.communication_manager.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        self.event_sdk.log_event_started("comm_c2s", event_edge_id=0)

        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.client_real_id, receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        sys_name = platform.system()
        if sys_name == "Darwin":
            sys_name = "MacOS"
        # Debug for simulation mobile system
        if self.args.is_mobile == 1:
            sys_name = MyMessage.MSG_CLIENT_OS_ANDROID

        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, sys_name)

        self.communication_manager.send_message(message)
        log_round_end(self.rank, self.round_idx)

        # Report client model to MLOps
        model_url = message.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS_URL)
        model_info = {
            "run_id": self.args.run_id,
            "edge_id": self.client_real_id,
            "round_idx": self.round_idx+1,
            "client_model_s3_address": model_url
        }
        self.mlops_logger.report_client_model_info(model_info)

    def send_client_status(self, receive_id, status="ONLINE"):
        logging.info("send_client_status")
        message = Message(MyMessage.MSG_TYPE_C2S_CLIENT_STATUS, self.client_real_id, receive_id)
        sys_name = platform.system()
        if sys_name == "Darwin":
            sys_name = "MacOS"
        # Debug for simulation mobile system
        if self.args.is_mobile == 1:
            sys_name = MyMessage.MSG_CLIENT_OS_ANDROID

        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_STATUS, status)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, sys_name)
        self.communication_manager.send_message(message)
        log_round_end(self.rank, self.round_idx)

    def report_training_status(self, status):
        self.mlops_logger.report_client_training_status(
            self.client_real_id, status
        )

    def report_sys_performances(self):
        train_acc = 1.0
        train_loss = 393.0
        while self.round_idx != self.num_rounds - 1:
            # Notify MLOps with system information.
            self.mlops_logger.report_system_metric()

            # train_acc -= 0.01
            # train_loss -= 2.0
            # train_metric = {"run_id": self.args.run_id, "timestamp": time.time(),
            #                 "accuracy": train_acc, "loss": train_loss}
            # self.mlops_logger.report_server_training_metric(train_metric)

            # model_out_file = "s3://fedmls3/02472215-8295-4ceb-a40e-1c1a2fb5d350"
            # model_info = {"run_id": self.args.run_id, "edge_id": self.client_real_id,
            #               "round_idx": self.round_idx, "client_model_s3_address": model_out_file}
            # self.mlops_logger.report_client_model_info(model_info)
            #
            # model_out_file = "s3://fedmls3/02472215-8295-4ceb-a40e-1c1a2fb5d350"
            # model_info = {"run_id": self.args.run_id, "round_index": self.round_idx,
            #               "global_aggregated_model_s3_address": model_out_file}
            # self.mlops_logger.report_aggregated_model_info(model_info)

            time.sleep(30)

    def run(self):
        # TODO: move registration to communication manager
        self.register_message_receive_handlers()
        self.communication_manager.run()


