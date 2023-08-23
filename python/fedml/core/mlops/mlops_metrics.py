import argparse
import json
import logging
import os
import time
import traceback
import uuid

import multiprocess as multiprocessing

from ...cli.edge_deployment.client_constants import ClientConstants
from ...cli.server_deployment.server_constants import ServerConstants
from ...core.distributed.communication.mqtt.mqtt_manager import MqttManager
from ...core.mlops.mlops_status import MLOpsStatus
from ...core.mlops.system_stats import SysStats
from .mlops_utils import MLOpsUtils


class MLOpsMetrics(object):
    FEDML_SYS_PERF_RUNNING_FILE_NAME = "sys_perf.id"

    def __new__(cls, *args, **kw):
        if not hasattr(cls, "_instance"):
            orig = super(MLOpsMetrics, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
            cls._instance.init()
        return cls._instance

    def __init__(self):
        pass

    def init(self):
        self.sys_stats_process = None
        self.messenger = None
        self.args = None
        self.run_id = None
        self.edge_id = None
        self.server_agent_id = None
        self.sys_performances = None
        self.sys_perf_event = None
        self.current_device_status = ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE
        self.current_run_status = ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED

    def set_messenger(self, msg_messenger, args=None):
        self.messenger = msg_messenger
        if args is not None:
            self.args = args
            self.run_id = args.run_id
            if args.role == "server":
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

    def report_client_training_status(self, edge_id, status, running_json=None, is_from_model=False, in_run_id=None):
        run_id = 0
        if self.run_id is not None:
            run_id = self.run_id

        if in_run_id is not None:
            run_id = in_run_id

        self.common_report_client_training_status(edge_id, status)

        self.common_report_client_id_status(run_id, edge_id, status)

        self.report_client_device_status_to_web_ui(edge_id, status)

        if is_from_model:
            from ...cli.model_deployment.device_client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance().save_job(run_id, edge_id, status, running_json)
        else:
            from ...cli.edge_deployment.client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance().save_job(run_id, edge_id, status, running_json)

    def report_client_device_status_to_web_ui(self, edge_id, status):
        """
        this is used for notifying the client device status to MLOps Frontend
        """
        if status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE:
            return

        run_id = 0
        if self.run_id is not None:
            run_id = self.run_id
        topic_name = "fl_client/mlops/status"
        msg = {"edge_id": edge_id, "run_id": run_id, "status": status, "version": "v1.0"}
        message_json = json.dumps(msg)
        logging.info("report_client_device_status. message_json = %s" % message_json)
        MLOpsStatus.get_instance().set_client_status(edge_id, status)
        self.messenger.send_message_json(topic_name, message_json)

    def common_report_client_training_status(self, edge_id, status):
        # if not self.comm_sanity_check():
        #     logging.info("comm_sanity_check at report_client_training_status.")
        #     return
        """
        this is used for notifying the client status to MLOps (both FedML CLI and backend can consume it)
        """
        run_id = 0
        if self.run_id is not None:
            run_id = self.run_id
        topic_name = "fl_run/fl_client/mlops/status"
        msg = {"edge_id": edge_id, "run_id": run_id, "status": status}
        message_json = json.dumps(msg)
        # logging.info("report_client_training_status. message_json = %s" % message_json)
        MLOpsStatus.get_instance().set_client_status(edge_id, status)
        self.messenger.send_message_json(topic_name, message_json)

    def broadcast_client_training_status(self, edge_id, status, is_from_model=False):
        # if not self.comm_sanity_check():
        #     return
        """
        this is used for broadcasting the client status to MLOps (backend can consume it)
        """
        run_id = 0
        if self.run_id is not None:
            run_id = self.run_id

        self.common_broadcast_client_training_status(edge_id, status)

        self.report_client_device_status_to_web_ui(edge_id, status)
        if is_from_model:
            from ...cli.model_deployment.device_client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance().save_job(run_id, edge_id, status)
        else:
            from ...cli.edge_deployment.client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance().save_job(run_id, edge_id, status)

    def common_broadcast_client_training_status(self, edge_id, status):
        # if not self.comm_sanity_check():
        #     return
        """
        this is used for broadcasting the client status to MLOps (backend can consume it)
        """
        run_id = 0
        if self.run_id is not None:
            run_id = self.run_id
        topic_name = "fl_run/fl_client/mlops/status"
        msg = {"edge_id": edge_id, "run_id": run_id, "status": status}
        message_json = json.dumps(msg)
        logging.info("report_client_training_status. message_json = %s" % message_json)
        self.messenger.send_message_json(topic_name, message_json)

    def client_send_exit_train_msg(self, run_id, edge_id, status, msg=None):
        topic_exit_train_with_exception = "flserver_agent/" + str(run_id) + "/client_exit_train_with_exception"
        msg = {"run_id": run_id, "edge_id": edge_id, "status": status, "msg": msg if msg is not None else ""}
        message_json = json.dumps(msg)
        logging.info("client_send_exit_train_msg.")
        self.messenger.send_message_json(topic_exit_train_with_exception, message_json)

    def report_client_id_status(self, run_id, edge_id, status, running_json=None, is_from_model=False):
        # if not self.comm_sanity_check():
        #     return
        """
        this is used for communication between client agent (FedML cli module) and client
        """
        self.common_report_client_id_status(run_id, edge_id, status)

        self.report_client_device_status_to_web_ui(edge_id, status)

        if is_from_model:
            from ...cli.model_deployment.device_client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance().save_job(run_id, edge_id, status, running_json)
        else:
            from ...cli.edge_deployment.client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance().save_job(run_id, edge_id, status, running_json)

    def common_report_client_id_status(self, run_id, edge_id, status):
        # if not self.comm_sanity_check():
        #     return
        """
        this is used for communication between client agent (FedML cli module) and client
        """
        topic_name = "fl_client/flclient_agent_" + str(edge_id) + "/status"
        msg = {"run_id": run_id, "edge_id": edge_id, "status": status}
        message_json = json.dumps(msg)
        # logging.info("report_client_id_status. message_json = %s" % message_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_server_training_status(self, run_id, status, role=None, running_json=None, is_from_model=False):
        # if not self.comm_sanity_check():
        #     return
        self.common_report_server_training_status(run_id, status, role)

        self.report_server_device_status_to_web_ui(run_id, status, role)

        if is_from_model:
            from ...cli.model_deployment.device_server_data_interface import FedMLServerDataInterface
            FedMLServerDataInterface.get_instance().save_job(run_id, self.edge_id, status, running_json)
        else:
            from ...cli.server_deployment.server_data_interface import FedMLServerDataInterface
            FedMLServerDataInterface.get_instance().save_job(run_id, self.edge_id, status, running_json)

    def report_server_device_status_to_web_ui(self, run_id, status, role=None):
        """
        this is used for notifying the server device status to MLOps Frontend
        """
        if status == ServerConstants.MSG_MLOPS_DEVICE_STATUS_IDLE:
            return

        topic_name = "fl_server/mlops/status"
        if role is None:
            role = "normal"
        msg = {
            "run_id": run_id,
            "edge_id": self.edge_id,
            "status": status,
            "role": role,
            "version": "v1.0"
        }
        logging.info("report_server_device_status. msg = %s" % msg)
        message_json = json.dumps(msg)
        MLOpsStatus.get_instance().set_server_status(self.edge_id, status)
        self.messenger.send_message_json(topic_name, message_json)

    def common_report_server_training_status(self, run_id, status, role=None):
        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_run/fl_server/mlops/status"
        if role is None:
            role = "normal"
        msg = {
            "run_id": run_id,
            "edge_id": self.edge_id,
            "status": status,
            "role": role,
        }
        # logging.info("report_server_training_status. msg = %s" % msg)
        message_json = json.dumps(msg)
        MLOpsStatus.get_instance().set_server_status(self.edge_id, status)
        self.messenger.send_message_json(topic_name, message_json)
        self.report_server_id_status(run_id, status)

    def broadcast_server_training_status(self, run_id, status, role=None, is_from_model=False):
        if self.messenger is None:
            return
        topic_name = "fl_run/fl_server/mlops/status"
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

        self.report_server_device_status_to_web_ui(run_id, status, role)

        if is_from_model:
            from ...cli.model_deployment.device_server_data_interface import FedMLServerDataInterface
            FedMLServerDataInterface.get_instance().save_job(run_id, self.edge_id, status)
        else:
            from ...cli.server_deployment.server_data_interface import FedMLServerDataInterface
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
        self.messenger.send_message_json(topic_name, message_json)

        self.report_server_device_status_to_web_ui(run_id, status)

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

    def report_llm_record(self, metric_json):
        # if not self.comm_sanity_check():
        #     return
        topic_name = "model_serving/mlops/llm_input_output_record"
        logging.info("report_llm_record. message_json = %s" % metric_json)
        message_json = json.dumps(metric_json)
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

            current_time_ms = MLOpsUtils.get_ntp_time()
            if current_time_ms is None:
                current_time = int(time.time()*1000)
            else:
                current_time = int(current_time_ms)

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
                "timestamp": int(current_time)
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

    def stop_sys_perf(self):
        if self.sys_perf_event is not None:
            self.sys_perf_event.set()

    def setup_sys_perf_process(self, sys_args):
        self.stop_sys_perf()

        self.args = sys_args
        if self.sys_perf_event is None:
            self.sys_perf_event = multiprocessing.Event()
        self.sys_perf_event.clear()

        self.sys_stats_process = multiprocessing.Process(target=self.report_sys_performances,
                                                         args=(self.sys_perf_event,))
        self.sys_stats_process.start()

    @staticmethod
    def report_sys_perf(sys_args):
        metrics = MLOpsMetrics()
        metrics.setup_sys_perf_process(sys_args)

    def report_sys_performances(self, sys_event):
        self.sys_perf_event = sys_event
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
        while not self.should_stop_sys_perf():
            try:
                self.report_system_metric()
            except Exception as e:
                logging.debug("exception when reporting system pref: {}.".format(traceback.format_exc()))
                pass

            time.sleep(10)

        logging.info("System metrics process is about to exit.")
        mqtt_mgr.loop_stop()
        mqtt_mgr.disconnect()

    def report_json_message(self, topic, payload):
        self.messenger.send_message_json(topic, payload)

    def should_stop_sys_perf(self):
        if self.sys_perf_event is not None and self.sys_perf_event.is_set():
            return True

        return False


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
        sys_metrics.stop_sys_perf()
        break
    pass
