
import json
import logging
import os
import time
import traceback
from abc import ABC, abstractmethod

import fedml
from ..comm_utils.constants import SchedulerConstants
from ..comm_utils.job_utils import JobRunnerUtils, DockerArgs
from ..comm_utils.run_process_utils import RunProcessUtils
from ....core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from ....core.mlops.mlops_configs import MLOpsConfigs
from ....core.mlops.mlops_runtime_log_daemon import MLOpsRuntimeLogDaemon
from ..comm_utils import sys_utils
from ....core.mlops.mlops_utils import MLOpsUtils
from ..scheduler_core.compute_cache_manager import ComputeCacheManager
from ..scheduler_core.ota_upgrade import FedMLOtaUpgrade
from .client_data_interface import FedMLClientDataInterface
from ..scheduler_core.scheduler_base_protocol_manager import FedMLSchedulerBaseProtocolManager
from ..scheduler_core.general_constants import GeneralConstants


class FedMLBaseSlaveProtocolManager(FedMLSchedulerBaseProtocolManager, ABC):

    def __init__(self, args, agent_config=None):
        FedMLSchedulerBaseProtocolManager.__init__(self, args, agent_config=agent_config)

        self.request_json = None
        self.disable_client_login = None
        self.args = args
        self.message_status_runner = None
        self.message_center = None
        self.status_center = None
        self.run_id = None
        self.edge_id = args.edge_id
        self.general_edge_id = None
        self.edge_user_name = args.user_name
        self.edge_extra_url = args.extra_url
        self.server_agent_id = args.edge_id
        self.current_device_id = args.current_device_id
        self.unique_device_id = args.unique_device_id
        self.agent_config = agent_config
        self.topic_start_train = None
        self.topic_report_status = None
        self.topic_ota_msg = None
        self.topic_request_device_info = None
        self.topic_request_device_info_from_mlops = None
        self.topic_client_logout = None
        self.topic_response_job_status = None
        self.topic_report_device_status_in_job = None
        self.fl_topic_start_train = None
        self.fl_topic_request_device_info = None
        self.communication_mgr = None
        self.subscribed_topics = list()
        self.ota_upgrade = FedMLOtaUpgrade(edge_id=args.edge_id)
        self.running_request_json = dict()
        self.start_request_json = None
        self.user_name = args.user_name
        self.general_edge_id = args.general_edge_id
        self.server_id = args.server_id
        self.model_device_server_id = None
        self.model_device_client_edge_id_list = None

    @abstractmethod
    def generate_topics(self):
        # The MQTT message topic format is as follows: <sender>/<receiver>/<action>

        # The topic for stopping training
        self.topic_start_train = "flserver_agent/" + str(self.edge_id) + "/start_train"

        # The topic for reporting current device status.
        self.topic_report_status = "mlops/report_device_status"

        # The topic for OTA messages from the MLOps.
        self.topic_ota_msg = "mlops/flclient_agent_" + str(self.edge_id) + "/ota"

        # The topic for requesting device info from the client.
        self.topic_request_device_info = "server/client/request_device_info/" + str(self.edge_id)

        # The topic for requesting device info from mlops.
        self.topic_request_device_info_from_mlops = f"deploy/mlops/slave_agent/request_device_info/{self.edge_id}"

        # The topic for requesting device info from MLOps.
        self.topic_client_logout = "mlops/client/logout/" + str(self.edge_id)

        # The topic for getting job status from the status center.
        self.topic_response_job_status = f"master_agent/somewhere/response_job_status/{self.edge_id}"

        # The topic for getting device status of job from the status center.
        self.topic_report_device_status_in_job = f"slave_job/slave_agent/report_device_status_in_job"

        # The topic for reporting online status
        self.topic_active = "flclient_agent/active"

        # The topic for last-will messages.
        self.topic_last_will = "flclient_agent/last_will_msg"

        if self.general_edge_id is not None:
            self.fl_topic_start_train = "flserver_agent/" + str(self.general_edge_id) + "/start_train"
            self.fl_topic_request_device_info = "server/client/request_device_info/" + str(self.general_edge_id)

        # Subscribe topics for starting train, stopping train and fetching client status.
        self.subscribed_topics.clear()
        self.add_subscribe_topic(self.topic_start_train)
        self.add_subscribe_topic(self.topic_report_status)
        self.add_subscribe_topic(self.topic_ota_msg)
        self.add_subscribe_topic(self.topic_request_device_info)
        self.add_subscribe_topic(self.topic_request_device_info_from_mlops)
        self.add_subscribe_topic(self.topic_client_logout)
        self.add_subscribe_topic(self.topic_response_job_status)
        self.add_subscribe_topic(self.topic_report_device_status_in_job)
        if self.general_edge_id is not None:
            self.add_subscribe_topic(self.fl_topic_start_train)
            self.add_subscribe_topic(self.fl_topic_request_device_info)

    @abstractmethod
    def add_protocol_handler(self):
        # Add the message listeners for all topics, the following is an example.
        # self.add_message_listener(self.topic_start_train, self.callback_start_train)
        # Add the message listeners for all topics
        self.add_message_listener(self.topic_start_train, self.callback_start_train)
        self.add_message_listener(self.topic_ota_msg, FedMLBaseSlaveProtocolManager.callback_client_ota_msg)
        self.add_message_listener(self.topic_report_status, self.callback_report_current_status)
        self.add_message_listener(self.topic_request_device_info, self.callback_report_device_info)
        self.add_message_listener(self.topic_request_device_info_from_mlops, self.callback_request_device_info_from_mlops)
        self.add_message_listener(self.topic_client_logout, self.callback_client_logout)
        self.add_message_listener(self.topic_response_job_status, self.callback_response_job_status)
        self.add_message_listener(self.topic_report_device_status_in_job, self.callback_response_device_status_in_job)
        self.add_message_listener(self.fl_topic_start_train, self.callback_start_train)
        self.add_message_listener(self.fl_topic_request_device_info, self.callback_report_device_info)

    @abstractmethod
    def _get_job_runner_manager(self):
        return None

    @abstractmethod
    def _init_extra_items(self):
        os.environ["FEDML_CURRENT_EDGE_ID"] = str(self.edge_id)
        if not ComputeCacheManager.get_instance().set_redis_params():
            os.environ["FEDML_DISABLE_REDIS_CONNECTION"] = "1"

    def add_subscribe_topic(self, topic):
        self.subscribed_topics.append(topic)

    def stop(self):
        if self.model_device_client_edge_id_list is not None:
            self.model_device_client_edge_id_list.clear()
            self.model_device_client_edge_id_list = None

        super().stop()

    def on_agent_communication_connected(self, mqtt_client_object):
        super().on_agent_communication_connected(mqtt_client_object)

        self._process_connection_ready()

        payload = {"model_master_device_id": self.model_device_server_id,
                   "model_slave_device_id_list": self.model_device_client_edge_id_list}
        self.receive_message(self.topic_request_device_info, json.dumps(payload))

    def on_agent_communication_disconnected(self, mqtt_client_object):
        super().on_agent_communication_disconnected(mqtt_client_object)

        self._process_connection_lost()

    @abstractmethod
    def _process_connection_ready(self):
        pass

    @abstractmethod
    def _process_connection_lost(self):
        pass

    def print_connected_info(self):
        print("\nCongratulations, your device is connected to the FedML MLOps platform successfully!")
        print(f"Your FedML Edge ID is {str(self.edge_id)}, unique device ID is {str(self.unique_device_id)}, "
              f"master deploy ID is {str(self.model_device_server_id)}, "
              f"worker deploy ID is {self.model_device_client_edge_id_list}"
              )
        if self.edge_extra_url is not None and self.edge_extra_url != "":
            print(f"You may visit the following url to fill in more information with your device.\n"
                  f"{self.edge_extra_url}")

    def callback_start_train(self, topic, payload):
        # Parse the parameters
        request_json = json.loads(payload)
        is_retain = request_json.get("is_retain", False)
        if is_retain:
            return
        run_id = request_json["runId"]
        edge_id = str(topic).split("/")[-2]
        self.args.run_id = run_id
        self.args.edge_id = edge_id

        # Start log processor for current run
        MLOpsRuntimeLog.get_instance(self.args).init_logs(log_level=logging.INFO)
        MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(
            run_id, edge_id, log_source=SchedulerConstants.get_log_source(request_json))
        logging.info("start the log processor")

        # Fetch the config
        try:
            MLOpsConfigs.fetch_all_configs()
        except Exception as e:
            logging.error(f"Failed to fetch all configs with Exception {e}. Traceback: {traceback.format_exc()}")
            pass

        # Check if the slave agent is disabled.
        if not FedMLClientDataInterface.get_instance().get_agent_status():
            request_json = json.loads(payload)
            run_id = request_json["runId"]
            logging.error(
                "FedMLDebug - Receive: topic ({}), payload ({}), but the client agent is disabled. {}".format(
                    topic, payload, traceback.format_exc()
                )
            )
            # Send failed msg when exceptions.
            self.status_reporter.report_client_id_status(
                edge_id, GeneralConstants.MSG_MLOPS_CLIENT_STATUS_EXCEPTION, run_id=run_id,
                msg=f"the client agent {edge_id} is disabled")
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, edge_id)
            return

        # Print the payload
        logging.info(
            f"FedMLDebug - run id {run_id}, Receive at callback_start_train: topic ({topic}), payload ({payload})"
        )

        # Occupy GPUs
        server_agent_id = request_json["cloud_agent_id"]
        scheduler_match_info = request_json.get("scheduler_match_info", {})
        matched_gpu_num = scheduler_match_info.get("matched_gpu_num", 0)
        model_master_device_id = scheduler_match_info.get("model_master_device_id", None)
        model_slave_device_id = scheduler_match_info.get("model_slave_device_id", None)
        model_slave_device_id_list = scheduler_match_info.get("model_slave_device_id_list", None)
        run_config = request_json.get("run_config", {})
        run_params = run_config.get("parameters", {})
        serving_args = run_params.get("serving_args", {})
        endpoint_id = serving_args.get("endpoint_id", None)
        job_yaml = run_params.get("job_yaml", {})
        job_type = job_yaml.get("job_type", SchedulerConstants.JOB_TASK_TYPE_TRAIN)
        cuda_visible_gpu_ids_str = None
        if not (job_type == SchedulerConstants.JOB_TASK_TYPE_SERVE or
                job_type == SchedulerConstants.JOB_TASK_TYPE_DEPLOY):
            cuda_visible_gpu_ids_str = JobRunnerUtils.get_instance().occupy_gpu_ids(
                run_id, matched_gpu_num, edge_id, inner_id=endpoint_id,
                model_master_device_id=model_master_device_id,
                model_slave_device_id=model_slave_device_id)
        else:
            # Save the relationship between run id and endpoint
            ComputeCacheManager.get_instance().set_redis_params()
            ComputeCacheManager.get_instance().get_gpu_cache().set_endpoint_run_id_map(
                endpoint_id, run_id)

            # Report the run status with finished status and return
            self.generate_status_report(run_id, edge_id, server_agent_id=server_agent_id).report_client_id_status(
                edge_id, GeneralConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED, run_id=run_id)

            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, edge_id)
            return
        logging.info(
            f"Run started, available gpu ids: {JobRunnerUtils.get_instance().get_available_gpu_id_list(edge_id)}")

        # Set the listener for job status from master agent
        self.setup_listener_job_status(run_id)

        # Start server with multiprocessing mode
        self.request_json = request_json
        run_id_str = str(run_id)
        self.running_request_json[run_id_str] = request_json
        self._get_job_runner_manager().start_job_runner(
            run_id, request_json, args=self.args, edge_id=edge_id,
            sender_message_queue=self.message_center.get_sender_message_queue(),
            listener_message_queue=self.get_listener_message_queue(),
            status_center_queue=self.get_status_queue(),
            cuda_visible_gpu_ids_str=cuda_visible_gpu_ids_str,
            process_name=GeneralConstants.get_launch_slave_job_process_name(run_id, edge_id)
        )
        run_process = self._get_job_runner_manager().get_runner_process(run_id)
        if run_process is not None:
            GeneralConstants.save_run_process(run_id, run_process.pid)

        # Register the job launch message into the status center
        self.register_job_launch_message(topic, payload)

    def callback_report_current_status(self, topic, payload):
        logging.info(
            f"FedMLDebug - Receive: topic ({topic}), payload ({payload})"
        )

        self.send_agent_active_msg(self.edge_id)
        if self.general_edge_id is not None:
            self.send_agent_active_msg(self.general_edge_id)

    @staticmethod
    def callback_client_ota_msg(topic, payload):
        logging.info(
            f"FedMLDebug - Receive: topic ({topic}), payload ({payload})"
        )

        request_json = json.loads(payload)
        cmd = request_json["cmd"]

        if cmd == GeneralConstants.FEDML_OTA_CMD_UPGRADE:
            FedMLOtaUpgrade.process_ota_upgrade_msg()
            # Process(target=FedMLClientRunner.process_ota_upgrade_msg).start()
            raise Exception("After upgraded, restart runner...")
        elif cmd == GeneralConstants.FEDML_OTA_CMD_RESTART:
            raise Exception("Restart runner...")

    def callback_report_device_info(self, topic, payload):
        payload_json = json.loads(payload)
        server_id = payload_json.get("server_id", 0)
        run_id = payload_json.get("run_id", 0)
        listen_edge_id = str(topic).split("/")[-1]
        context = payload_json.get("context", None)
        need_gpu_info = payload_json.get("need_gpu_info", False)
        need_running_process_list = payload_json.get("need_running_process_list", False)
        model_master_device_id = payload_json.get("model_master_device_id", None)
        model_slave_device_id_list = payload_json.get("model_slave_device_id_list", None)
        if model_master_device_id is not None:
            self.model_device_server_id = model_master_device_id
        if model_slave_device_id_list is not None:
            self.model_device_client_edge_id_list = model_slave_device_id_list
        response_topic = f"client/server/response_device_info/{server_id}"
        if self.mlops_metrics is not None:
            if not need_gpu_info:
                device_info_json = {
                    "edge_id": listen_edge_id,
                    "fedml_version": fedml.__version__,
                    "user_id": self.args.user_name
                }
            else:
                total_mem, free_mem, total_disk_size, free_disk_size, cup_utilization, cpu_cores, gpu_cores_total, \
                    gpu_cores_available, sent_bytes, recv_bytes, gpu_available_ids = sys_utils.get_sys_realtime_stats()
                host_ip = sys_utils.get_host_ip()
                host_port = sys_utils.get_available_port()
                gpu_available_ids = JobRunnerUtils.get_available_gpu_id_list(self.edge_id)
                gpu_available_ids = JobRunnerUtils.trim_unavailable_gpu_ids(gpu_available_ids)
                gpu_cores_available = len(gpu_available_ids)
                gpu_list = sys_utils.get_gpu_list()
                device_info_json = {
                    "edge_id": listen_edge_id,
                    "memoryTotal": round(total_mem * MLOpsUtils.BYTES_TO_GB, 2),
                    "memoryAvailable": round(free_mem * MLOpsUtils.BYTES_TO_GB, 2),
                    "diskSpaceTotal": round(total_disk_size * MLOpsUtils.BYTES_TO_GB, 2),
                    "diskSpaceAvailable": round(free_disk_size * MLOpsUtils.BYTES_TO_GB, 2),
                    "cpuUtilization": round(cup_utilization, 2),
                    "cpuCores": cpu_cores,
                    "gpuCoresTotal": gpu_cores_total,
                    "gpuCoresAvailable": gpu_cores_available,
                    "gpu_available_ids": gpu_available_ids,
                    "gpu_list": gpu_list,
                    "node_ip": host_ip,
                    "node_port": host_port,
                    "networkTraffic": sent_bytes + recv_bytes,
                    "updateTime": int(MLOpsUtils.get_ntp_time()),
                    "fedml_version": fedml.__version__,
                    "user_id": self.args.user_name
                }
            if need_running_process_list:
                device_info_json["run_process_list_map"] = self.get_all_run_process_list_map()
            salve_device_ids = list()
            if self.model_device_client_edge_id_list is not None and \
                    isinstance(self.model_device_client_edge_id_list, list):
                for model_client_edge_id in self.model_device_client_edge_id_list:
                    salve_device_ids.append(model_client_edge_id)
            response_payload = {"slave_device_id": None if len(salve_device_ids) <= 0 else salve_device_ids[0],
                                "slave_device_id_list": salve_device_ids,
                                "master_device_id": self.model_device_server_id,
                                "run_id": run_id, "edge_id": listen_edge_id,
                                "edge_info": device_info_json}
            if context is not None:
                response_payload["context"] = context
            self.message_center.send_message(response_topic, json.dumps(response_payload), run_id=run_id)

    def callback_request_device_info_from_mlops(self, topic, payload):
        self.response_device_info_to_mlops(topic, payload)

    def response_device_info_to_mlops(self, topic, payload):
        response_topic = f"deploy/slave_agent/mlops/response_device_info"
        response_payload = {"run_id": self.run_id, "slave_agent_device_id": self.edge_id,
                            "fedml_version": fedml.__version__, "edge_id": self.edge_id}
        self.message_center.send_message(response_topic, json.dumps(response_payload))

    def callback_client_logout(self, topic, payload):
        payload_json = json.loads(payload)
        secret = payload_json.get("auth", None)
        if secret is None or str(secret) != "246b1be6-0eeb-4b17-b118-7d74de1975d4":
            return
        logging.info("Received the logout request.")
        self._get_job_runner_manager().stop_all_job_runner()
        self.disable_client_login = True
        time.sleep(3)
        os.system("fedml logout")

    def callback_response_device_status_in_job(self, topic, payload):
        # Parse the parameters
        payload_json = json.loads(payload)
        run_id = payload_json.get("run_id", None)
        job_status = payload_json.get("status", None)
        edge_id = payload_json.get("edge_id", None)

        # process the status
        logging.info("process status in the device status callback.")
        self.process_status(run_id, job_status, edge_id)

    def callback_response_job_status(self, topic, payload):
        # Parse the parameters
        payload_json = json.loads(payload)
        run_id = payload_json.get("run_id", None)
        master_agent = payload_json.get("master_agent", None)
        job_status = payload_json.get("job_status", None)
        fedml_version = payload_json.get("fedml_version", None)
        edge_id = payload_json.get("edge_id", None)

        # process the status
        logging.info("process status in the job status callback.")
        self.process_status(run_id, job_status, edge_id, master_id=master_agent)

    def callback_broadcasted_job_status(self, topic, payload):
        # Parse the parameters
        payload_json = json.loads(payload)
        run_id = payload_json.get("run_id", None)
        job_status = payload_json.get("status", None)

        # process the status
        logging.info("process status in the broadcast job status callback.")
        self.process_status(run_id, job_status, self.edge_id)

    def generate_protocol_manager(self):
        message_status_runner = self._generate_protocol_manager_instance(
            self.args, agent_config=self.agent_config
        )
        message_status_runner.request_json = self.request_json
        message_status_runner.disable_client_login = self.disable_client_login
        message_status_runner.message_center_name = self.message_center_name
        message_status_runner.run_id = self.run_id
        message_status_runner.edge_id = self.edge_id
        message_status_runner.edge_user_name = self.edge_user_name
        message_status_runner.edge_extra_url = self.edge_extra_url
        message_status_runner.server_agent_id = self.server_agent_id
        message_status_runner.current_device_id = self.current_device_id
        message_status_runner.unique_device_id = self.unique_device_id
        message_status_runner.subscribed_topics = self.subscribed_topics
        message_status_runner.running_request_json = self.running_request_json
        message_status_runner.request_json = self.start_request_json
        message_status_runner.user_name = self.user_name
        message_status_runner.general_edge_id = self.general_edge_id
        message_status_runner.server_id = self.server_id
        message_status_runner.model_device_server_id = self.model_device_server_id
        message_status_runner.model_device_client_edge_id_list = self.model_device_client_edge_id_list
        message_status_runner.status_queue = self.get_status_queue()

        return message_status_runner

    def process_status(self, run_id, status, edge_id, master_id=None):
        run_id_str = str(run_id)

        # Process the completed status
        if status == GeneralConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED or \
                status == GeneralConstants.MSG_MLOPS_CLIENT_STATUS_FAILED or \
                status == GeneralConstants.MSG_MLOPS_CLIENT_STATUS_KILLED:
            self._get_job_runner_manager().complete_job_runner(run_id)

            # Stop the sys perf process
            # noinspection PyBoardException
            try:
                self.mlops_metrics.stop_sys_perf()
            except Exception as ex:
                logging.error(f"Failed to stop sys perf with Exception {ex}. Traceback: {traceback.format_exc()}")
                pass

            # Stop the user process
            try:
                GeneralConstants.cleanup_learning_process(run_id)
                GeneralConstants.cleanup_bootstrap_process(run_id)
                GeneralConstants.cleanup_run_process(run_id)
            except Exception as e:
                logging.error(
                    f"Failed to cleanup run when finished with Exception {e}. Traceback: {traceback.format_exc()}")
                pass

            # Get the running json.
            running_json = self.running_request_json.get(run_id_str)
            if running_json is None:
                try:
                    current_job = FedMLClientDataInterface.get_instance().get_job_by_id(run_id)
                    running_json = json.loads(current_job.running_json)
                except Exception as e:
                    logging.error(f"Failed to get running json with Exception {e}. Traceback: {traceback.format_exc()}")

            # Cleanup the containers and release the gpu ids.
            if running_json is not None:
                job_type = JobRunnerUtils.parse_job_type(running_json)
                if not SchedulerConstants.is_deploy_job(job_type):
                    logging.info(f"[run/device][{run_id}/{edge_id}] Release gpu resource when run ended.")
                    self._get_job_runner_manager().cleanup_containers_and_release_gpus(run_id, edge_id, job_type)

            # Stop the runner process
            run_process = self._get_job_runner_manager().get_runner_process(run_id)
            if run_process is not None:
                if run_process.pid is not None:
                    RunProcessUtils.kill_process(run_process.pid)

                    # Terminate the run docker container if exists
                    try:
                        container_name = JobRunnerUtils.get_run_container_name(run_id)
                        docker_client = JobRunnerUtils.get_docker_client(DockerArgs())
                        logging.info(f"Terminating the run docker container {container_name} if exists...")
                        JobRunnerUtils.remove_run_container_if_exists(container_name, docker_client)
                    except Exception as e:
                        logging.error(f"Error occurred when terminating docker container."
                                      f"Exception: {e}, Traceback: {traceback.format_exc()}.")

            # Stop log processor for current run
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, edge_id)

    def setup_listener_job_status(self, run_id):
        # Setup MQTT message listener to receive the job status from master agent;
        topic_job_status_from_master = f"master_agent/slave_agent/job_status/{run_id}"
        self.add_message_listener(topic_job_status_from_master, self.callback_broadcasted_job_status)
        self.subscribe_msg(topic_job_status_from_master)

    def remove_listener_job_status(self, run_id):
        # Remove MQTT message listener from master agent;
        topic_job_status_from_master = f"master_agent/slave_agent/job_status/{run_id}"
        self.remove_message_listener(topic_job_status_from_master)
        self.unsubscribe_msg(topic_job_status_from_master)

    def get_all_run_process_list_map(self):
        run_process_dict = dict()
        all_runner_pid_dict = self._get_job_runner_manager().get_all_runner_pid_map()
        if all_runner_pid_dict is None:
            return run_process_dict
        for run_id_str, process in all_runner_pid_dict.items():
            cur_run_process_list = GeneralConstants.get_learning_process_list(run_id_str)
            run_process_dict[run_id_str] = cur_run_process_list

        return run_process_dict

    def stop_job(self, run_id):
        self._get_job_runner_manager().stop_job_runner(run_id)

    @staticmethod
    def get_start_train_topic_with_edge_id(edge_id):
        return "flserver_agent/" + str(edge_id) + "/start_train"

    @abstractmethod
    def _generate_protocol_manager_instance(self, args, agent_config=None):
        return None
