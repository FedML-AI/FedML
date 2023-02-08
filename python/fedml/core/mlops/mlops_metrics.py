import argparse
import json
import logging
import os
import time
import uuid

import multiprocess as multiprocessing

from ..common.singleton import Singleton
from ...cli.edge_deployment.client_constants import ClientConstants
from ...cli.server_deployment.server_constants import ServerConstants
from ...core.distributed.communication.mqtt.mqtt_manager import MqttManager
from ...core.mlops.mlops_status import MLOpsStatus
from ...core.mlops.system_stats import SysStats
from ...cli.edge_deployment.client_data_interface import FedMLClientDataInterface
from ...cli.server_deployment.server_data_interface import FedMLServerDataInterface


class MLOpsMetrics(Singleton):
    FEDML_SYS_PERF_RUNNING_FILE_NAME = "sys_perf.id"

    def __init__(self):
        self.messenger = None
        self.args = None
        self.run_id = None
        self.edge_id = None
        self.server_agent_id = None
        self.sys_performances = None
        self.is_sys_perf_reporting = False
        self.current_device_status = ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE
        self.current_run_status = ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED
        self.sys_perf_running_file = os.path.join(
            ClientConstants.get_data_dir(),
            ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
            MLOpsMetrics.FEDML_SYS_PERF_RUNNING_FILE_NAME,
        )

    def set_messenger(self, msg_messenger, args=None):
        self.messenger = msg_messenger
        if args is not None:
            self.args = args
            self.run_id = args.run_id
            if args.rank == 0:
                if hasattr(args, "server_id"):
                    self.edge_id = args.server_id
                else:
                    self.edge_id = 0

                self.sys_perf_running_file = os.path.join(
                    ServerConstants.get_data_dir(),
                    ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME,
                    MLOpsMetrics.FEDML_SYS_PERF_RUNNING_FILE_NAME,
                )
            else:
                if hasattr(args, "client_id"):
                    self.edge_id = args.client_id
                elif hasattr(args, "client_id_list"):
                    edge_ids = json.loads(args.client_id_list)
                    if len(edge_ids) > 0:
                        self.edge_id = edge_ids[0]
                    else:
                        self.edge_id = 0
                else:
                    self.edge_id = 0

                self.sys_perf_running_file = os.path.join(
                    ClientConstants.get_data_dir(),
                    ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
                    MLOpsMetrics.FEDML_SYS_PERF_RUNNING_FILE_NAME,
                )

            if hasattr(args, "server_agent_id"):
                self.server_agent_id = args.server_agent_id
            else:
                self.server_agent_id = self.edge_id

    def comm_sanity_check(self):
        if self.messenger is None:
            logging.info("self.messenger is Null")
            return False
        else:
            return True

    def report_client_training_status(self, edge_id, status, running_json=None):
        # if not self.comm_sanity_check():
        #     logging.info("comm_sanity_check at report_client_training_status.")
        #     return
        """
        this is used for notifying the client status to MLOps (both web UI, FedML CLI and backend can consume it)
        """
        run_id = 0
        if self.run_id is not None:
            run_id = self.run_id
        topic_name = "fl_client/mlops/status"
        msg = {"edge_id": edge_id, "run_id": run_id, "status": status}
        message_json = json.dumps(msg)
        logging.info("report_client_training_status. message_json = %s" % message_json)
        MLOpsStatus.get_instance().set_client_status(edge_id, status)
        self.messenger.send_message_json(topic_name, message_json)
        self.report_client_id_status(run_id, edge_id, status)

        if status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED and run_id != 0:
            self.report_server_training_status(run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED,
                                               call_from_server=False)
            self.client_send_stop_train_msg()

        FedMLClientDataInterface.get_instance().save_job(run_id, edge_id, status, running_json)

    def broadcast_client_training_status(self, edge_id, status, should_send_stop_msg=True):
        # if not self.comm_sanity_check():
        #     return
        """
        this is used for broadcasting the client status to MLOps (both web UI and backend can consume it)
        """
        run_id = 0
        if self.run_id is not None:
            run_id = self.run_id
        topic_name = "fl_client/mlops/status"
        msg = {"edge_id": edge_id, "run_id": run_id, "status": status}
        message_json = json.dumps(msg)
        logging.info("report_client_training_status. message_json = %s" % message_json)
        self.messenger.send_message_json(topic_name, message_json)

        if status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED and run_id != 0:
            self.report_server_training_status(run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED,
                                               call_from_server=False)
            if should_send_stop_msg:
                self.client_send_stop_train_msg()

        FedMLClientDataInterface.get_instance().save_job(run_id, edge_id, status)

    def client_send_stop_train_msg(self):
        current_job = FedMLClientDataInterface.get_instance().get_current_job()
        if current_job is None or current_job.running_json == "":
            return
        running_json_obj = json.loads(current_job.running_json)
        server_id = running_json_obj.get("server_id", None)
        if server_id is None:
            return

        topic_stop_train = "mlops/flserver_agent_" + str(server_id) + "/stop_train"
        message_json = current_job.running_json
        logging.info("send_stop_train_msg.")
        self.messenger.send_message_json(topic_stop_train, message_json)

    def server_send_stop_train_msg(self):
        current_job = FedMLServerDataInterface.get_instance().get_current_job()
        if current_job is None or current_job.running_json == "":
            return
        running_json_obj = json.loads(current_job.running_json)
        server_id = running_json_obj.get("server_id", None)
        if server_id is None:
            return

        topic_stop_train = "mlops/flserver_agent_" + str(server_id) + "/stop_train"
        message_json = current_job.running_json
        logging.info("send_stop_train_msg.")
        self.messenger.send_message_json(topic_stop_train, message_json)

    def report_client_id_status(self, run_id, edge_id, status):
        # if not self.comm_sanity_check():
        #     return
        """
        this is used for communication between client agent (FedML cli module) and client
        """
        topic_name = "fl_client/flclient_agent_" + str(edge_id) + "/status"
        msg = {"run_id": run_id, "edge_id": edge_id, "status": status}
        message_json = json.dumps(msg)
        logging.info("report_client_id_status. message_json = %s" % message_json)
        MLOpsStatus.get_instance().set_client_agent_status(self.edge_id, status)
        self.messenger.send_message_json(topic_name, message_json)

    def report_server_training_status(self, run_id, status, role=None, running_json=None, call_from_server=True):
        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_server/mlops/status"
        if role is None:
            role = "normal"
        msg = {
            "run_id": run_id,
            "edge_id": self.edge_id,
            "status": status,
            "role": role,
        }
        logging.info("report_server_training_status. msg = %s" % msg)
        message_json = json.dumps(msg)
        MLOpsStatus.get_instance().set_server_status(self.edge_id, status)
        self.messenger.send_message_json(topic_name, message_json)
        self.report_server_id_status(run_id, status)

        if call_from_server:
            if status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED and run_id != 0:
                self.server_send_stop_train_msg()

            FedMLServerDataInterface.get_instance().save_job(run_id, self.edge_id, status, running_json)

    def broadcast_server_training_status(self, run_id, status, role=None):
        if self.messenger is None:
            return
        topic_name = "fl_server/mlops/status"
        if role is None:
            role = "normal"
        msg = {
            "run_id": run_id,
            "edge_id": self.edge_id,
            "status": status,
            "role": role,
        }
        logging.info("broadcast_server_training_status. msg = %s" % msg)
        message_json = json.dumps(msg)
        self.messenger.send_message_json(topic_name, message_json)

        if status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED and run_id != 0:
            self.server_send_stop_train_msg()

        FedMLServerDataInterface.get_instance().save_job(run_id, self.edge_id, status)

    def report_server_id_status(self, run_id, status):
        # if not self.comm_sanity_check():
        #     return
        server_agent_id = self.server_agent_id
        topic_name = "fl_server/flserver_agent_" + str(server_agent_id) + "/status"
        msg = {"run_id": run_id, "edge_id": self.edge_id, "status": status}
        message_json = json.dumps(msg)
        # logging.info("report_server_id_status server id {}".format(server_agent_id))
        # logging.info("report_server_id_status. message_json = %s" % message_json)
        MLOpsStatus.get_instance().set_server_agent_status(server_agent_id, status)
        self.messenger.send_message_json(topic_name, message_json)

    def report_client_training_metric(self, metric_json):
        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_client/mlops/training_metrics"
        logging.info("report_client_training_metric. message_json = %s" % metric_json)
        message_json = json.dumps(metric_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_server_training_metric(self, metric_json):
        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_server/mlops/training_progress_and_eval"
        logging.info("report_server_training_metric. message_json = %s" % metric_json)
        message_json = json.dumps(metric_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_server_training_round_info(self, round_info):
        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_server/mlops/training_roundx"
        message_json = json.dumps(round_info)
        self.messenger.send_message_json(topic_name, message_json)

    def report_client_model_info(self, model_info_json):
        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_server/mlops/client_model"
        message_json = json.dumps(model_info_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_aggregated_model_info(self, model_info_json):
        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_server/mlops/global_aggregated_model"
        message_json = json.dumps(model_info_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_training_model_net_info(self, model_net_info_json):
        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_server/mlops/training_model_net"
        message_json = json.dumps(model_net_info_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_system_metric(self, metric_json=None):
        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_client/mlops/system_performance"
        if metric_json is None:
            if self.sys_performances is None:
                self.sys_performances = SysStats()
            if self.sys_performances is None:
                return

            self.sys_performances.produce_info()
            metric_json = {
                "run_id": self.run_id,
                "edge_id": self.edge_id,
                "cpu_utilization": round(
                    self.sys_performances.get_cpu_utilization(), 4
                ),
                "SystemMemoryUtilization": round(
                    self.sys_performances.get_system_memory_utilization(), 4
                ),
                "process_memory_in_use": round(
                    self.sys_performances.get_process_memory_in_use(), 4
                ),
                "process_memory_in_use_size": round(
                    self.sys_performances.get_process_memory_in_use_size(), 4
                ),
                "process_memory_available": round(
                    self.sys_performances.get_process_memory_available(), 4
                ),
                "process_cpu_threads_in_use": round(
                    self.sys_performances.get_process_cpu_threads_in_use(), 4
                ),
                "disk_utilization": round(
                    self.sys_performances.get_disk_utilization(), 4
                ),
                "network_traffic": round(
                    self.sys_performances.get_network_traffic(), 4
                ),
                "gpu_utilization": round(
                    self.sys_performances.get_gpu_utilization(), 4
                ),
                "gpu_temp": round(self.sys_performances.get_gpu_temp(), 4),
                "gpu_time_spent_accessing_memory": round(
                    self.sys_performances.get_gpu_time_spent_accessing_memory(), 4
                ),
                "gpu_memory_allocated": round(
                    self.sys_performances.get_gpu_memory_allocated(), 4
                ),
                "gpu_power_usage": round(
                    self.sys_performances.get_gpu_power_usage(), 4
                ),
            }
        message_json = json.dumps(metric_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_logs_updated(self, run_id):
        # if not self.comm_sanity_check():
        #     return
        topic_name = "mlops/runtime_logs/" + str(run_id)
        msg = {"time": time.time()}
        message_json = json.dumps(msg)
        logging.info("report_logs_updated. message_json = %s" % message_json)
        self.messenger.send_message_json(topic_name, message_json)

    def set_sys_reporting_status(self, enable, is_client=True):
        if is_client:
            self.sys_perf_running_file = os.path.join(
                ClientConstants.get_data_dir(),
                ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
                MLOpsMetrics.FEDML_SYS_PERF_RUNNING_FILE_NAME,
            )
        else:
            self.sys_perf_running_file = os.path.join(
                ServerConstants.get_data_dir(),
                ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME,
                MLOpsMetrics.FEDML_SYS_PERF_RUNNING_FILE_NAME,
            )
        self.is_sys_perf_reporting = enable
        sys_perf_file_handle = open(self.sys_perf_running_file, "w")
        if sys_perf_file_handle is not None:
            sys_perf_file_handle.writelines([str(self.is_sys_perf_reporting)])
            sys_perf_file_handle.flush()
            sys_perf_file_handle.close()

    def is_system_perf_reporting(self):
        sys_perf_file_handle = open(self.sys_perf_running_file, "r")
        if sys_perf_file_handle is not None:
            self.is_sys_perf_reporting = eval(sys_perf_file_handle.readline())
            sys_perf_file_handle.close()
        return self.is_sys_perf_reporting

    @staticmethod
    def report_sys_perf(sys_args, is_client=True):
        sys_metrics = MLOpsMetrics()
        sys_metrics.args = sys_args
        sys_metrics.set_sys_reporting_status(True, is_client)
        sys_metrics.is_system_perf_reporting()
        sys_metrics.sys_stats_process = multiprocessing.Process(
            target=sys_metrics.report_sys_performances
        )
        sys_metrics.sys_stats_process.start()

    def report_sys_performances(self):
        self.set_messenger(None, self.args)
        mqtt_mgr = MqttManager(
            self.args.mqtt_config_path["BROKER_HOST"],
            self.args.mqtt_config_path["BROKER_PORT"],
            self.args.mqtt_config_path["MQTT_USER"],
            self.args.mqtt_config_path["MQTT_PWD"],
            180,
            "FedML_Metrics_SysPerf_{}_{}_{}".format(str(self.args.device_id), str(self.edge_id), str(uuid.uuid4()))
        )

        self.set_messenger(mqtt_mgr, self.args)
        mqtt_mgr.connect()
        mqtt_mgr.loop_start()

        # Notify MLOps with system information.
        while self.is_system_perf_reporting() is True:
            try:
                self.report_system_metric()
            except Exception as e:
                pass

            time.sleep(10)

        logging.info("System metrics process is about to exit.")
        mqtt_mgr.loop_stop()
        mqtt_mgr.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run_id", "-r", help="run id")
    parser.add_argument("--client_id", "-c", help="client id")
    parser.add_argument("--server_id", "-s", help="server id")
    args = parser.parse_args()
    mqtt_config = dict()
    mqtt_config["BROKER_HOST"] = "127.0.0.1"
    mqtt_config["BROKER_PORT"] = 1883
    mqtt_config["MQTT_USER"] = "admin"
    mqtt_config["MQTT_PWD"] = "sdfdf"
    setattr(args, "mqtt_config_path", mqtt_config)
    if args.client_id is not None:
        setattr(args, "rank", 1)
    else:
        setattr(args, "rank", 0)

    MLOpsMetrics.report_sys_perf(args)
    while True:
        time.sleep(5)
        sys_metrics = MLOpsMetrics()
        sys_metrics.set_sys_reporting_status(False)
        break
    pass
