
import json
import logging
import os
import time

from ...computing.scheduler.slave.client_constants import ClientConstants
from ...computing.scheduler.master.server_constants import ServerConstants
from ...core.mlops.mlops_status import MLOpsStatus
from .mlops_job_perfs import MLOpsJobPerfStats
from .mlops_device_perfs import MLOpsDevicePerfStats


class MLOpsMetrics(object):
    def __new__(cls, *args, **kw):
        """
        Create a singleton instance of MLOpsMetrics.

        Args:
            cls: The class.
            *args: Variable-length argument list.
            **kw: Keyword arguments.

        Returns:
            MLOpsMetrics: The MLOpsMetrics instance.
        """
        if not hasattr(cls, "_instance"):
            orig = super(MLOpsMetrics, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
            cls._instance.init()
        return cls._instance

    def __init__(self):
        """
        Initialize the MLOpsMetrics object.
        """

        pass

    def init(self):
        """
        Initialize the MLOpsMetrics object attributes.
        """
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
        """
        Set the messenger for communication.

        Args:
            msg_messenger: The message messenger.
            args: The system arguments.
        """
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
        """
        Check if communication is set up properly.

        Returns:
            bool: True if communication is set up, otherwise False.
        """
        if self.messenger is None:
            logging.info("self.messenger is Null")
            return False
        else:
            return True

    def report_client_training_status(self, edge_id, status, running_json=None, is_from_model=False, in_run_id=None):
        """
        Report client training status to various components.

        Args:
            edge_id: The ID of the edge device.
            status: The status of the training.
            running_json: The running JSON information.
            is_from_model: Whether the report is from the model.
            in_run_id: The run ID.
        """
        run_id = 0
        if self.run_id is not None:
            run_id = self.run_id

        if in_run_id is not None:
            run_id = in_run_id

        self.common_report_client_training_status(edge_id, status)

        self.common_report_client_id_status(run_id, edge_id, status)

        self.report_client_device_status_to_web_ui(edge_id, status)

        if is_from_model:
            from ...computing.scheduler.model_scheduler.device_client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance().save_job(
                run_id, edge_id, status, running_json)
        else:
            from ...computing.scheduler.slave.client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance().save_job(
                run_id, edge_id, status, running_json)

    def report_client_device_status_to_web_ui(self, edge_id, status):
        """
        Report the client device status to MLOps Frontend.

        Args:
            edge_id (int): The ID of the edge device.
            status (str): The status of the client device.
        """
        if status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE:
            return

        run_id = 0
        if self.run_id is not None:
            run_id = self.run_id
        topic_name = "fl_client/mlops/status"
        msg = {"edge_id": edge_id, "run_id": run_id,
               "status": status, "version": "v1.0"}
        message_json = json.dumps(msg)
        logging.info(
            "report_client_device_status. message_json = %s" % message_json)
        MLOpsStatus.get_instance().set_client_status(edge_id, status)
        self.messenger.send_message_json(topic_name, message_json)

    def common_report_client_training_status(self, edge_id, status):
        # if not self.comm_sanity_check():
        #     logging.info("comm_sanity_check at report_client_training_status.")
        #     return
        """
        Common method for reporting client training status to MLOps.

        Args:
            edge_id (int): The ID of the edge device.
            status (str): The status of the client device.
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
        Broadcast client training status to MLOps.

        Args:
            edge_id (int): The ID of the edge device.
            status (str): The status of the client device.
            is_from_model (bool): Whether the report is from the model.
        """
        run_id = 0
        if self.run_id is not None:
            run_id = self.run_id

        self.common_broadcast_client_training_status(edge_id, status)

        self.report_client_device_status_to_web_ui(edge_id, status)
        if is_from_model:
            from ...computing.scheduler.model_scheduler.device_client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance().save_job(run_id, edge_id, status)
        else:
            from ...computing.scheduler.slave.client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance().save_job(run_id, edge_id, status)

    def common_broadcast_client_training_status(self, edge_id, status):
        # if not self.comm_sanity_check():
        #     return
        """
        Common method for broadcasting client training status to MLOps.

        Args:
            edge_id (int): The ID of the edge device.
            status (str): The status of the client device.
        """
        run_id = 0
        if self.run_id is not None:
            run_id = self.run_id
        topic_name = "fl_run/fl_client/mlops/status"
        msg = {"edge_id": edge_id, "run_id": run_id, "status": status}
        message_json = json.dumps(msg)
        logging.info(
            "report_client_training_status. message_json = %s" % message_json)
        self.messenger.send_message_json(topic_name, message_json)

    def client_send_exit_train_msg(self, run_id, edge_id, status, msg=None):
        """
        Send an exit train message for a client.

        Args:
            run_id (int): The ID of the training run.
            edge_id (int): The ID of the edge device.
            status (str): The status of the client.
            msg (str, optional): Additional message (default is None).
        """
        topic_exit_train_with_exception = "flserver_agent/" + \
            str(run_id) + "/client_exit_train_with_exception"
        msg = {"run_id": run_id, "edge_id": edge_id,
               "status": status, "msg": msg if msg is not None else ""}
        message_json = json.dumps(msg)
        logging.info("client_send_exit_train_msg.")
        self.messenger.send_message_json(
            topic_exit_train_with_exception, message_json)

    def report_client_id_status(self, run_id, edge_id, status, running_json=None,
                                is_from_model=False, server_id="0"):
        # if not self.comm_sanity_check():
        #     return
        """
        Report client ID status to MLOps.

        Args:
            run_id (int): The ID of the training run.
            edge_id (int): The ID of the edge device.
            status (str): The status of the client.
            running_json: JSON information about the running state (default is None).
            is_from_model (bool): Whether the report is from the model (default is False).
            server_id (str): The ID of the server (default is "0").
        """
        self.common_report_client_id_status(run_id, edge_id, status, server_id)

        self.report_client_device_status_to_web_ui(edge_id, status)

        if is_from_model:
            from ...computing.scheduler.model_scheduler.device_client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance().save_job(
                run_id, edge_id, status, running_json)
        else:
            from ...computing.scheduler.slave.client_data_interface import FedMLClientDataInterface
            FedMLClientDataInterface.get_instance().save_job(
                run_id, edge_id, status, running_json)

    def common_report_client_id_status(self, run_id, edge_id, status, server_id="0"):
        # if not self.comm_sanity_check():
        #     return
        """
        Common method for reporting client ID status to MLOps.

        Args:
            run_id (int): The ID of the training run.
            edge_id (int): The ID of the edge device.
            status (str): The status of the client device.
            server_id (str): The ID of the server (default is "0").
        """
        topic_name = "fl_client/flclient_agent_" + str(edge_id) + "/status"
        msg = {"run_id": run_id, "edge_id": edge_id,
               "status": status, "server_id": server_id}
        message_json = json.dumps(msg)
        # logging.info("report_client_id_status. message_json = %s" % message_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_server_training_status(self, run_id, status, role=None, running_json=None, is_from_model=False):
        """
        Report server training status to MLOps.

        Args:
            run_id (int): The ID of the training run.
            status (str): The status of the server.
            role (str, optional): The role of the server (default is None).
            running_json: JSON information about the running state (default is None).
            is_from_model (bool): Whether the report is from the model (default is False).
        """
        # if not self.comm_sanity_check():
        #     return
        self.common_report_server_training_status(run_id, status, role)

        self.report_server_device_status_to_web_ui(run_id, status, role)

        if is_from_model:
            from ...computing.scheduler.model_scheduler.device_server_data_interface import FedMLServerDataInterface
            FedMLServerDataInterface.get_instance().save_job(
                run_id, self.edge_id, status, running_json)
        else:
            from ...computing.scheduler.master.server_data_interface import FedMLServerDataInterface
            FedMLServerDataInterface.get_instance().save_job(
                run_id, self.edge_id, status, running_json)

    def report_server_device_status_to_web_ui(self, run_id, status, role=None):
        """
        Report server device status to MLOps Frontend.

        Args:
            run_id (int): The ID of the training run.
            status (str): The status of the server device.
            role (str, optional): The role of the server (default is None).
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
        """
        Common method for reporting server training status to MLOps.

        Args:
            run_id (int): The ID of the training run.
            status (str): The status of the server.
            role (str, optional): The role of the server (default is None).
        """
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

    def broadcast_server_training_status(self, run_id, status, role=None, is_from_model=False, edge_id=None):
        """
        Broadcast server training status to MLOps.

        Args:
            run_id (int): The ID of the training run.
            status (str): The status of the server.
            role (str, optional): The role of the server (default is None).
            is_from_model (bool): Whether the report is from the model (default is False).
            edge_id (int, optional): The ID of the edge device (default is None).
        """
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

        self.report_server_device_status_to_web_ui(run_id, status, role)

        if is_from_model:
            from ...computing.scheduler.model_scheduler.device_server_data_interface import FedMLServerDataInterface
            FedMLServerDataInterface.get_instance().save_job(run_id, self.edge_id, status)
        else:
            from ...computing.scheduler.master.server_data_interface import FedMLServerDataInterface
            FedMLServerDataInterface.get_instance().save_job(run_id, self.edge_id, status)

    def report_server_id_status(self, run_id, status, edge_id=None, server_id=None, server_agent_id=None):
        """
        Report server ID status to MLOps.

        Args:
            run_id (int): The ID of the training run.
            status (str): The status of the server.
            edge_id (int, optional): The ID of the edge device (default is None).
            server_id (str, optional): The ID of the server (default is None).
            server_agent_id (int, optional): The ID of the server agent (default is None).
        """
        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_server/flserver_agent_" + str(server_agent_id if server_agent_id is not None else
                                                       self.server_agent_id) + "/status"
        msg = {"run_id": run_id,
               "edge_id": edge_id if edge_id is not None else self.edge_id, "status": status}
        if server_id is not None:
            msg["server_id"] = server_id
        message_json = json.dumps(msg)
        # logging.info("report_server_id_status server id {}".format(server_agent_id))
        logging.info("report_server_id_status. message_json = %s" %
                     message_json)
        self.messenger.send_message_json(topic_name, message_json)

        self.report_server_device_status_to_web_ui(run_id, status)

    def report_client_training_metric(self, metric_json):
        """
        Report client training metrics to MLOps.

        Args:
            metric_json (dict): JSON containing client training metrics.
        """

        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_client/mlops/training_metrics"
        logging.info(
            "report_client_training_metric. message_json = %s" % metric_json)
        message_json = json.dumps(metric_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_server_training_metric(self, metric_json):
        """
        Report server training metrics to MLOps.

        Args:
            metric_json (dict): JSON containing server training metrics.
        """
        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_server/mlops/training_progress_and_eval"
        logging.info(
            "report_server_training_metric. message_json = %s" % metric_json)
        message_json = json.dumps(metric_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_server_training_round_info(self, round_info):
        """
        Report server training round information to MLOps.

        Args:
            round_info (dict): JSON containing server training round information.
        """

        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_server/mlops/training_roundx"
        message_json = json.dumps(round_info)
        self.messenger.send_message_json(topic_name, message_json)

    def report_client_model_info(self, model_info_json):
        """
        Report client model information to MLOps.

        Args:
            model_info_json (dict): JSON containing client model information.
        """
        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_server/mlops/client_model"
        message_json = json.dumps(model_info_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_aggregated_model_info(self, model_info_json):
        """
        Report aggregated model information to MLOps.

        Args:
            model_info_json (dict): JSON containing aggregated model information.
        """

        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_server/mlops/global_aggregated_model"
        message_json = json.dumps(model_info_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_training_model_net_info(self, model_net_info_json):
        """
        Report training model network information to MLOps.

        Args:
            model_net_info_json (dict): JSON containing training model network information.
        """
        # if not self.comm_sanity_check():
        #     return
        topic_name = "fl_server/mlops/training_model_net"
        message_json = json.dumps(model_net_info_json)
        self.messenger.send_message_json(topic_name, message_json)

    def report_llm_record(self, metric_json):
        """
        Report low-latency model (LLM) input-output record to MLOps.

        Args:
            metric_json (dict): JSON containing low-latency model input-output record.
        """
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
        Report the computing cost of a job running on an edge to MLOps.

        Args:
            job_id (str): The ID of the job.
            edge_id (str): The ID of the edge device.
            computing_started_time (float): The timestamp when computing started.
            computing_ended_time (float): The timestamp when computing ended.
            user_id (str): The user ID.
            api_key (str): The API key.
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
        """
        Report that runtime logs have been updated to MLOps.

        Args:
            run_id (int): The ID of the training run.
        """
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
        """
        Report artifact information to MLOps.

        Args:
            job_id (str): The ID of the job associated with the artifact.
            edge_id (str): The ID of the edge device where the artifact is generated.
            artifact_name (str): The name of the artifact.
            artifact_type (str): The type of the artifact.
            artifact_local_path (str): The local path to the artifact.
            artifact_url (str): The URL of the artifact.
            artifact_ext_info (dict): Additional information about the artifact.
            artifact_desc (str): A description of the artifact.
            timestamp (float): The timestamp when the artifact was generated.
        """
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

    def report_sys_perf(self, sys_args, mqtt_config):
        """
        Report system performance metrics to MLOps.

        Args:
            sys_args (object): System arguments object containing performance metrics.
            mqtt_config (str): Path to the MQTT configuration.
        """
        setattr(sys_args, "mqtt_config_path", mqtt_config)
        run_id = getattr(sys_args, "run_id", 0)
        self.fl_job_perf.add_job(run_id, os.getpid())
        self.fl_job_perf.report_job_stats(sys_args)

    def stop_sys_perf(self):
        """
        Stop reporting system performance metrics to MLOps.
        """
        self.fl_job_perf.stop_job_stats()

    def report_job_perf(self, sys_args, mqtt_config, job_process_id):
        """
        Report job performance metrics to MLOps.

        Args:
            sys_args (object): System arguments object containing job performance metrics.
            mqtt_config (str): Path to the MQTT configuration.
            job_process_id (int): The process ID of the job.
        """
        setattr(sys_args, "mqtt_config_path", mqtt_config)
        run_id = getattr(sys_args, "run_id", 0)
        self.job_perfs.add_job(run_id, job_process_id)
        self.job_perfs.report_job_stats(sys_args)

    def stop_job_perf(self):
        """
        Stop reporting job performance metrics to MLOps.
        """
        self.job_perfs.stop_job_stats()

    def report_device_realtime_perf(self, sys_args, mqtt_config):
        """
        Report real-time device performance metrics to MLOps.

        Args:
            sys_args (object): System arguments object containing real-time device performance metrics.
            mqtt_config (str): Path to the MQTT configuration.
        """
        setattr(sys_args, "mqtt_config_path", mqtt_config)
        self.device_perfs.report_device_realtime_stats(sys_args)

    def stop_device_realtime_perf(self):
        """
        Stop reporting real-time device performance metrics to MLOps.
        """

        self.device_perfs.stop_device_realtime_stats()

    def report_json_message(self, topic, payload):
        """
        Report a JSON message to a specified topic.

        Args:
            topic (str): The MQTT topic to publish the message to.
            payload (dict): The JSON payload to be sent.
        """
        self.messenger.send_message_json(topic, payload)
