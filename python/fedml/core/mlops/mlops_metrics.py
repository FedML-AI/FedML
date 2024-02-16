
import json
import logging
import os
import time

import requests

import fedml
from . import MLOpsConfigs
from ...computing.scheduler.slave.client_constants import ClientConstants
from ...computing.scheduler.master.server_constants import ServerConstants
from ...core.mlops.mlops_status import MLOpsStatus
from .mlops_job_perfs import MLOpsJobPerfStats
from .mlops_device_perfs import MLOpsDevicePerfStats


class MLOpsMetrics(object):
    def __new__(cls, *args, **kw):
        if not hasattr(cls, "_instance"):
            orig = super(MLOpsMetrics, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
            cls._instance.init()
        return cls._instance

    def __init__(self):
        pass

    def init(self):
        self.messenger = None
        self.args = None
        self.run_id = None
        self.edge_id = None
        self.server_agent_id = None
        self.current_device_status = ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE
        self.current_run_status = ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED
        self.fl_job_perf = MLOpsJobPerfStats()
        self.job_perfs = MLOpsJobPerfStats()
        self.device_perfs = MLOpsDevicePerfStats()

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

    def report_client_training_status(self, edge_id, status, running_json=None, is_from_model=False, run_id=0):
        self.common_report_client_training_status(edge_id, status, run_id=run_id)

        if is_from_model:
            from ...computing.scheduler.model_scheduler.device_client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance().save_job(run_id, edge_id, status, running_json)
        else:
            from ...computing.scheduler.slave.client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance().save_job(run_id, edge_id, status, running_json)

    def report_client_device_status_to_web_ui(self, edge_id, status, run_id=0):
        """
        this is used for notifying the client device status to MLOps Frontend
        """
        if status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE:
            return

        topic_name = "fl_client/mlops/status"
        msg = {"edge_id": edge_id, "run_id": run_id, "status": status, "version": "v1.0"}
        message_json = json.dumps(msg)
        logging.info("report_client_device_status. message_json = %s" % message_json)
        MLOpsStatus.get_instance().set_client_status(edge_id, status)
        self.messenger.send_message_json(topic_name, message_json)

    def common_report_client_training_status(self, edge_id, status, run_id=0):
        # if not self.comm_sanity_check():
        #     logging.info("comm_sanity_check at report_client_training_status.")
        #     return
        """
        this is used for notifying the client status to MLOps (both FedML CLI and backend can consume it)
        Currently, the INITIALIZING, and RUNNING are report using this method.
        """
        topic_name = "fl_run/fl_client/mlops/status"
        msg = {"edge_id": edge_id, "run_id": run_id, "status": status}
        message_json = json.dumps(msg)
        logging.info("report_client_training_status. message_json = %s" % message_json)
        MLOpsStatus.get_instance().set_client_status(edge_id, status)
        self.messenger.send_message_json(topic_name, message_json)

    def broadcast_client_training_status(self, edge_id, status, is_from_model=False, run_id=0):
        # if not self.comm_sanity_check():
        #     return
        """
        this is used for broadcasting the client status to MLOps (backend can consume it)
        """
        self.common_broadcast_client_training_status(edge_id, status, run_id=run_id)

        if is_from_model:
            from ...computing.scheduler.model_scheduler.device_client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance().save_job(run_id, edge_id, status)
        else:
            from ...computing.scheduler.slave.client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance().save_job(run_id, edge_id, status)

    def common_broadcast_client_training_status(self, edge_id, status, run_id=0):
        # if not self.comm_sanity_check():
        #     return
        """
        this is used for broadcasting the client status to MLOps (backend can consume it)
        Currently, the FINISHED are report using this method.
        """
        topic_name = "fl_run/fl_client/mlops/status"
        msg = {"edge_id": edge_id, "run_id": run_id, "status": status}
        message_json = json.dumps(msg)
        logging.info("broadcast_client_training_status. message_json = %s" % message_json)
        self.messenger.send_message_json(topic_name, message_json)

    def client_send_exit_train_msg(self, run_id, edge_id, status, msg=None):
        topic_exit_train_with_exception = "flserver_agent/" + str(run_id) + "/client_exit_train_with_exception"
        msg = {"run_id": run_id, "edge_id": edge_id, "status": status, "msg": msg if msg is not None else ""}
        message_json = json.dumps(msg)
        logging.info("client_send_exit_train_msg.")
        self.messenger.send_message_json(topic_exit_train_with_exception, message_json)

    def report_client_id_status(self, edge_id, status, running_json=None,
                                is_from_model=False, server_id="0", run_id=0, msg=""):
        # if not self.comm_sanity_check():
        #     return
        """
        this is used for communication between client agent (FedML cli module) and client
        """
        self.common_report_client_id_status(run_id, edge_id, status, server_id, msg=msg)

        if is_from_model:
            from ...computing.scheduler.model_scheduler.device_client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance().save_job(run_id, edge_id, status, running_json)
        else:
            from ...computing.scheduler.slave.client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance().save_job(run_id, edge_id, status, running_json)

    def common_report_client_id_status(self, run_id, edge_id, status, server_id="0", msg=""):
        # if not self.comm_sanity_check():
        #     return
        """
        this is used for communication between client agent (FedML cli module) and client
        """
        topic_name = "fl_client/flclient_agent_" + str(edge_id) + "/status"
        msg = {"run_id": run_id, "edge_id": edge_id, "status": status, "server_id": server_id, "msg": msg}
        message_json = json.dumps(msg)
        # logging.info("report_client_id_status. message_json = %s" % message_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_server_training_status(self, run_id, status, edge_id=0, role=None, running_json=None, is_from_model=False):
        # if not self.comm_sanity_check():
        #     return
        self.common_report_server_training_status(run_id, status, role=role, edge_id=edge_id)

        if is_from_model:
            from ...computing.scheduler.model_scheduler.device_server_data_interface import FedMLServerDataInterface
            FedMLServerDataInterface.get_instance().save_job(run_id, self.edge_id, status, running_json)
        else:
            from ...computing.scheduler.master.server_data_interface import FedMLServerDataInterface
            FedMLServerDataInterface.get_instance().save_job(run_id, self.edge_id, status, running_json)

    def report_server_device_status_to_web_ui(self, run_id, status, edge_id=0, role=None):
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
            "edge_id": edge_id,
            "status": status,
            "role": role,
            "version": "v1.0"
        }
        # logging.info("report_server_device_status. msg = %s" % msg)
        message_json = json.dumps(msg)
        MLOpsStatus.get_instance().set_server_status(self.edge_id, status)
        self.messenger.send_message_json(topic_name, message_json)

    def common_report_server_training_status(self, run_id, status, role=None, edge_id=0):
        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_run/fl_server/mlops/status"
        if role is None:
            role = "normal"
        msg = {
            "run_id": run_id,
            "edge_id": edge_id,
            "status": status,
            "role": role,
        }
        # logging.info("report_server_training_status. msg = %s" % msg)
        message_json = json.dumps(msg)
        MLOpsStatus.get_instance().set_server_status(self.edge_id, status)
        self.messenger.send_message_json(topic_name, message_json)

    def broadcast_server_training_status(self, run_id, status, role=None, is_from_model=False, edge_id=None):
        if self.messenger is None:
            return
        topic_name = "fl_run/fl_server/mlops/status"
        if role is None:
            role = "normal"
        msg = {
            "run_id": run_id,
            "edge_id": self.edge_id if edge_id is None else edge_id,
            "status": status,
            "role": role,
        }
        logging.info("broadcast_server_training_status. msg = %s" % msg)
        message_json = json.dumps(msg)
        self.messenger.send_message_json(topic_name, message_json)

        if is_from_model:
            from ...computing.scheduler.model_scheduler.device_server_data_interface import FedMLServerDataInterface
            FedMLServerDataInterface.get_instance().save_job(run_id, self.edge_id, status)
        else:
            from ...computing.scheduler.master.server_data_interface import FedMLServerDataInterface
            FedMLServerDataInterface.get_instance().save_job(run_id, self.edge_id, status)

    def report_server_id_status(self, run_id, status, edge_id=None, server_id=None, server_agent_id=None):
        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_server/flserver_agent_" + str(server_agent_id if server_agent_id is not None else
                                                       self.server_agent_id) + "/status"
        msg = {"run_id": run_id, "edge_id": edge_id if edge_id is not None else self.edge_id, "status": status}
        if server_id is not None:
            msg["server_id"] = server_id
        message_json = json.dumps(msg)
        logging.info(f"report_server_id_status; topic_name: {topic_name}, msg: {msg}")
        # logging.info("report_server_id_status server id {}".format(server_agent_id))
        # logging.info("report_server_id_status. message_json = %s" % message_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_client_training_metric(self, metric_json):
        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_client/mlops/training_metrics"
        logging.info("report_client_training_metric. message_json = %s" % metric_json)
        message_json = json.dumps(metric_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_server_training_metric(self, metric_json, payload=None):
        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_server/mlops/training_progress_and_eval"
        if payload is not None:
            message_json = payload
        else:
            message_json = json.dumps(metric_json)
        # logging.info("report_server_training_metric. message_json = %s" % metric_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_endpoint_metric(self, metric_json, payload=None):
        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_server/mlops/deploy_progress_and_eval"
        if payload is not None:
            message_json = payload
        else:
            message_json = json.dumps(metric_json)
        # logging.info("report_endpoint_metric. message_json = %s" % metric_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_fedml_train_metric(self, metric_json, run_id=0, is_endpoint=False):
        # if not self.comm_sanity_check():
        #     return
        topic_name = f"fedml_slave/fedml_master/metrics/{run_id}"
        logging.info("report_fedml_train_metric. message_json = %s" % metric_json)
        metric_json["is_endpoint"] = is_endpoint
        message_json = json.dumps(metric_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_fedml_run_logs(self, logs_json, run_id=0):
        # if not self.comm_sanity_check():
        #     return
        topic_name = f"fedml_slave/fedml_master/logs/{run_id}"
        message_json = json.dumps(logs_json)
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

    def report_edge_job_computing_cost(self, job_id, edge_id,
                                       computing_started_time, computing_ended_time,
                                       user_id, api_key):
        """
        this is used for reporting the computing cost of a job running on an edge to MLOps
        """
        topic_name = "ml_client/mlops/job_computing_cost"
        duration = computing_ended_time - computing_started_time
        if duration < 0:
            duration = 0
        msg = {"edge_id": edge_id, "job_id": job_id,
               "computing_started_time": computing_started_time,
               "computing_ended_time": computing_ended_time,
               "duration": duration, "user_id": user_id, "api_key": api_key}
        message_json = json.dumps(msg)
        self.messenger.send_message_json(topic_name, message_json)
        # logging.info("report_job_computing_cost. message_json = %s" % message_json)

    def report_logs_updated(self, run_id):
        # if not self.comm_sanity_check():
        #     return
        topic_name = "mlops/runtime_logs/" + str(run_id)
        msg = {"time": time.time()}
        message_json = json.dumps(msg)
        logging.info("report_logs_updated. message_json = %s" % message_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_artifact_info(self, job_id, edge_id, artifact_name, artifact_type,
                             artifact_local_path, artifact_url,
                             artifact_ext_info, artifact_desc,
                             timestamp):
        topic_name = "launch_device/mlops/artifacts"
        artifact_info_json = {
            "job_id": job_id,
            "edge_id": edge_id,
            "artifact_name": artifact_name,
            "artifact_local_path": artifact_local_path,
            "artifact_url": artifact_url,
            "artifact_type": artifact_type,
            "artifact_desc": artifact_desc,
            "artifact_ext_info": artifact_ext_info,
            "timestamp": timestamp
        }
        message_json = json.dumps(artifact_info_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_endpoint_status(self, end_point_id, model_status, timestamp=None,
                               end_point_name="", model_name="", model_inference_url=""):
        deployment_status_topic_prefix = "model_ops/model_device/return_deployment_status"
        deployment_status_topic = "{}/{}".format(deployment_status_topic_prefix, end_point_id)
        time_param = time.time_ns() / 1000.0 if timestamp is None else timestamp
        deployment_status_payload = {"end_point_id": end_point_id, "end_point_name": end_point_name,
                                     "model_name": model_name,
                                     "model_url": model_inference_url,
                                     "model_status": model_status,
                                     "timestamp": int(format(time_param, '.0f'))}

        self.messenger.send_message_json(deployment_status_topic, json.dumps(deployment_status_payload))
        self.messenger.send_message_json(deployment_status_topic_prefix, json.dumps(deployment_status_payload))

    def report_run_log(
            self, run_id, device_id, log_list, log_source=None, use_mqtt=False
    ):
        url = fedml._get_backend_service()
        log_sever_url = f"{url}/fedmlLogsServer/logs/update"
        log_upload_request = {
            "run_id": run_id,
            "edge_id": device_id,
            "logs": log_list,
            "create_time": time.time(),
            "update_time": time.time(),
            "created_by": str(device_id),
            "updated_by": str(device_id)
        }

        if log_source is not None and log_source != "":
            log_upload_request["source"] = log_source

        if use_mqtt:
            self.report_fedml_run_logs(log_upload_request, run_id=run_id)
        else:
            log_headers = {'Content-Type': 'application/json', 'Connection': 'close'}

            # send log data to the log server
            _, cert_path = MLOpsConfigs.get_request_params()
            if cert_path is not None:
                try:
                    requests.session().verify = cert_path
                    # logging.info(f"FedMLDebug POST log to server. run_id {run_id}, device_id {device_id}")
                    response = requests.post(
                        log_sever_url, json=log_upload_request, verify=True, headers=log_headers
                    )
                    # logging.info(f"FedMLDebug POST log to server run_id {run_id}, device_id {device_id}. response.status_code: {response.status_code}")

                except requests.exceptions.SSLError as err:
                    MLOpsConfigs.install_root_ca_file()
                    # logging.info(f"FedMLDebug POST log to server. run_id {run_id}, device_id {device_id}")
                    response = requests.post(
                        log_sever_url, json=log_upload_request, verify=True, headers=log_headers
                    )
                    # logging.info(f"FedMLDebug POST log to server run_id {run_id}, device_id {device_id}. response.status_code: {response.status_code}")
            else:
                # logging.info(f"FedMLDebug POST log to server. run_id {run_id}, device_id {device_id}")
                response = requests.post(log_sever_url, headers=log_headers, json=log_upload_request)
                # logging.info(f"FedMLDebug POST log to server. run_id {run_id}, device_id {device_id}. response.status_code: {response.status_code}")
            if response.status_code != 200:
                return

    def report_sys_perf(self, sys_args, mqtt_config, run_id=None, job_process_id=None):
        setattr(sys_args, "mqtt_config_path", mqtt_config)
        run_id = getattr(sys_args, "run_id", 0) if run_id is None else run_id
        setattr(sys_args, "run_id", run_id)
        self.fl_job_perf.add_job(run_id, os.getpid() if job_process_id is None else job_process_id)
        self.fl_job_perf.report_job_stats(sys_args)

    def stop_sys_perf(self):
        self.fl_job_perf.stop_job_stats()

    def report_job_perf(self, sys_args, mqtt_config, job_process_id):
        setattr(sys_args, "mqtt_config_path", mqtt_config)
        run_id = getattr(sys_args, "run_id", 0)
        self.job_perfs.add_job(run_id, job_process_id)
        self.job_perfs.report_job_stats(sys_args)

    def stop_job_perf(self):
        self.job_perfs.stop_job_stats()

    def report_device_realtime_perf(self, sys_args, mqtt_config, is_client=True):
        setattr(sys_args, "mqtt_config_path", mqtt_config)
        self.device_perfs.is_client = is_client
        self.device_perfs.report_device_realtime_stats(sys_args)

    def stop_device_realtime_perf(self):
        self.device_perfs.stop_device_realtime_stats()

    def report_json_message(self, topic, payload):
        self.messenger.send_message_json(topic, payload)