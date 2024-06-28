import json
import logging
import os
import shutil
import time
from os import listdir

from ....core.mlops.mlops_runtime_log_daemon import MLOpsRuntimeLogDaemon
from ....core.mlops.mlops_metrics import MLOpsMetrics
from ..slave.client_constants import ClientConstants
from ..master.server_constants import ServerConstants
from ..master.server_data_interface import FedMLServerDataInterface
from .message_common import LogArgs
from .general_constants import GeneralConstants
from ..scheduler_core.compute_cache_manager import ComputeCacheManager


class FedMLStatusManager(object):
    def __init__(self, run_id=None, edge_id=None, server_id=None,
                 edge_id_list=None, running_scheduler_contract=None,
                 status_center=None, message_center=None):
        self.run_id = run_id
        self.edge_id = edge_id
        self.server_id = server_id
        self.edge_id_list = edge_id_list
        self.edge_status_dict = None
        self.running_scheduler_contract = running_scheduler_contract if running_scheduler_contract is not None else dict()
        self.message_reporter = MLOpsMetrics()
        self.message_reporter.set_messenger(message_center)
        self.status_reporter = MLOpsMetrics()
        self.status_reporter.set_messenger(status_center, send_message_func=status_center.send_status_message)
        self.status_center = status_center
        self.message_center = message_center
        self.log_args = LogArgs(role="server", edge_id=self.edge_id,
                                server_id=self.server_id, log_file_dir=ServerConstants.get_log_file_dir())

        self.job_status_in_slave = dict()
        self.entire_job_status = None
        self.job_status_in_master = dict()
        self.slave_devices_status = dict()
        self.master_devices_status = dict()
        self.completed_job_status_list = [ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED,
                                          ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED,
                                          ServerConstants.MSG_MLOPS_SERVER_STATUS_KILLED]

    def __repr__(self):
        return "<{klass} @{id:x} {attrs}>".format(
            klass=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=" ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def add_job_status_in_slave(self, device_id, status):
        self.job_status_in_slave[device_id] = self._status_transition(status)

    def add_job_status_in_master(self, device_id, status):
        self.job_status_in_master[device_id] = self._status_transition(status)

    def set_entire_job_status(self, status):
        self.entire_job_status = status

    def add_slave_device_status(self, device_id, status):
        self.slave_devices_status[device_id] = self._status_transition(status)

    def add_master_device_status(self, run_id, device_id, status):
        self.master_devices_status[device_id] = self._status_transition(status)

    def get_job_status_in_slave(self, device_id):
        return self.job_status_in_slave.get(device_id, None)

    def get_job_status_in_master(self, device_id):
        return self.job_status_in_master.get(device_id, None)

    def get_entire_job_status(self):
        return self.entire_job_status

    def get_slave_device_status(self, device_id):
        return self.slave_devices_status.get(device_id, None)

    def get_master_device_status(self, device_id):
        return self.master_devices_status.get(device_id, None)

    def is_job_completed(self):
        if self.entire_job_status and self.entire_job_status in self.completed_job_status_list:
            return True
        return False

    def _status_transition(self, status):
        transition_status = status
        if self.entire_job_status is not None:
            if self.entire_job_status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED or \
                    self.entire_job_status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED:
                if status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED or \
                        status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED or \
                        status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED:
                    transition_status = status
                else:
                    transition_status = ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED

        return transition_status

    def save_job_status(self):
        ComputeCacheManager.get_instance().set_redis_params()
        ComputeCacheManager.get_instance().get_status_cache().save_job_status(
            self.run_id, self.get_entire_job_status())

    def save_device_status_in_job(self, device_id):
        ComputeCacheManager.get_instance().set_redis_params()
        ComputeCacheManager.get_instance().get_status_cache().save_device_status_in_job(
            self.run_id, device_id, self.get_job_status_in_slave(device_id))

    def process_job_completed_status(self, master_id, status):
        # Stop the system performance monitor
        try:
            self.message_reporter.stop_sys_perf()
        except Exception as ex:
            pass

        # Stop the job process
        ServerConstants.cleanup_learning_process(self.run_id)
        ServerConstants.cleanup_bootstrap_process(self.run_id)

        # Remove the package download directory.
        try:
            local_package_path = ServerConstants.get_package_download_dir()
            for package_file in listdir(local_package_path):
                if os.path.basename(package_file).startswith("run_" + str(self.run_id)):
                    shutil.rmtree(os.path.join(local_package_path, package_file), ignore_errors=True)
        except Exception as e:
            pass

        # Stop log processor for current run
        MLOpsRuntimeLogDaemon.get_instance(self.log_args).stop_log_processor(self.run_id, master_id)

        # RunProcessUtils.kill_process(cloud_server_process.pid)
        # self.stop_cloud_server()
        # self.remove_listener_for_run_metrics(self.run_id)
        # self.remove_listener_for_run_logs(self.run_id)

        self.message_center.receive_message(
            GeneralConstants.get_topic_complete_job(master_id),
            json.dumps(GeneralConstants.get_payload_complete_job(self.run_id, master_id)))

        if self.status_center.is_deployment_status_center:
            if status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED:
                self.report_deployment_status(self.run_id, GeneralConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED)

    def process_job_exception_status(self, master_id, status):
        # Report exception job status
        self.report_exception_status(status)

        # Save the job status to local storage
        FedMLServerDataInterface.get_instance().save_job_status(self.run_id, master_id, status, status)

    def process_job_running_status(self, master_id, status):
        self.message_reporter.report_server_training_status(
            self.run_id, status, edge_id=master_id, running_json=self.running_scheduler_contract, update_db=False)

    def status_center_process_master_status(self, topic, payload):
        request_json = json.loads(payload)
        is_retain = request_json.get("is_retain", False)
        if is_retain:
            return
        run_id = request_json["run_id"]
        status = request_json["status"]
        edge_id = request_json["edge_id"]
        server_id = request_json.get("server_id", None)
        if server_id is None or str(server_id) == "0":
            server_id = self.server_id
        run_id_str = str(run_id)

        # Process the job status
        if status in (ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED,
                      ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED,
                      ServerConstants.MSG_MLOPS_SERVER_STATUS_KILLED):
            self.process_job_completed_status(server_id, status)
        elif status == ServerConstants.MSG_MLOPS_SERVER_STATUS_EXCEPTION:
            self.process_job_exception_status(server_id, status)
        else:
            self.process_job_running_status(server_id, status)

        # Process the consensus status
        self.process_job_status_consensus(run_id, server_id, status)

    def process_job_status_consensus(self, run_id, master_id, status):
        # Set the master status in the job and entire job status
        self.set_entire_job_status(status)
        self.add_job_status_in_master(master_id, status)
        status = self.get_entire_job_status()

        # Set the device status based on the job status
        if self.edge_status_dict is not None:
            for edge_id_item, edge_status_item in self.edge_status_dict.items():
                if edge_id_item == "server":
                    continue

                # Calc the device status based on the job status
                consensus_device_status = FedMLStatusManager.get_device_consensus_status_in_job(
                    status, edge_status_item)
                if consensus_device_status is not None:
                    self.message_reporter.report_client_training_status(
                        edge_id_item, consensus_device_status, run_id=run_id, update_db=False)

        # Save the job status to local storage
        FedMLServerDataInterface.get_instance().save_job_status(run_id, master_id, status, status)

        # Report the status to message center
        self.message_reporter.report_server_training_status(run_id, status, edge_id=master_id, update_db=False)

        # Broadcast the status to slave agents
        self.message_reporter.report_job_status(run_id, status)

    @staticmethod
    def get_device_consensus_status_in_job(job_status, device_status):
        if job_status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED:
            if device_status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED or \
                    device_status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED or \
                    device_status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED:
                return device_status
            else:
                return ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED
        else:
            return None

    def get_device_consensus_status_in_current_device(self, edge_id, status):
        self.add_job_status_in_slave(edge_id, status)
        consensus_status = self.get_job_status_in_slave(edge_id)
        consensus_status = ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED \
            if consensus_status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_EXCEPTION else consensus_status
        return consensus_status

    def status_center_process_slave_status(self, topic, payload):
        payload_json = json.loads(payload)
        run_id = payload_json.get("run_id", None)
        edge_id = payload_json.get("edge_id", None)
        status = payload_json.get("status", None)
        init_edge_id_list = payload_json.get("init_all_edge_id_list", None)
        init_server_id = payload_json.get("init_server_id", None)

        if self.edge_status_dict is None:
            self.edge_status_dict = dict()

        if init_edge_id_list is not None:
            self.edge_status_dict[f"server"] = init_server_id
            for edge_id_item in init_edge_id_list:
                self.edge_status_dict[f"{edge_id_item}"] = \
                    ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE

        if run_id is not None and edge_id is not None:
            self.edge_status_dict[f"{edge_id}"] = status

            self.process_device_status(run_id, edge_id, status)

    def process_device_status(self, run_id, edge_id, status):
        number_of_failed_edges = 0
        number_of_finished_edges = 0
        number_of_killed_edges = 0
        server_id = self.edge_status_dict.get("server", 0)
        enable_fault_tolerance, fault_tolerance_rate = self.parse_fault_tolerance_params(run_id)
        running_edges_list = list()
        edge_nums = 0
        for edge_id_item, status_item in self.edge_status_dict.items():
            if edge_id_item == "server":
                continue

            edge_nums += 1
            if status_item is None or status_item == ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED or \
                    status_item == ServerConstants.MSG_MLOPS_SERVER_STATUS_EXCEPTION:
                number_of_failed_edges += 1
                continue

            if status_item == ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED:
                number_of_finished_edges += 1
                continue

            if status_item == ServerConstants.MSG_MLOPS_SERVER_STATUS_KILLED:
                number_of_killed_edges += 1
                continue

            if status_item == ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE or \
                    status_item == ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE:
                continue

            running_edges_list.append(edge_id_item)

        # Report client status
        consensus_status = self.get_device_consensus_status_in_current_device(edge_id, status)
        self.message_reporter.report_client_training_status(edge_id, consensus_status, run_id=run_id, update_db=False)

        # Report server status based on the fault tolerance model and parameters
        if edge_nums <= 0:
            return
        status_to_report = self.calculate_server_status(
            run_id, edge_nums, number_of_failed_edges, number_of_finished_edges, number_of_killed_edges,
            running_edges_list, enable_fault_tolerance=enable_fault_tolerance,
            fault_tolerance_rate=fault_tolerance_rate)
        if status_to_report is not None:
            logging.info(f"Run completed when processing edge status, will report status {status_to_report}")
            self.report_server_status(run_id, server_id, server_id, status_to_report)

    def calculate_server_status(
            self, run_id, total_edge_nums, number_of_failed_edges, number_of_finished_edges,
            number_of_killed_edges, running_edges_list, enable_fault_tolerance=False,
            fault_tolerance_rate=0.8
    ):
        # Report server status based on the fault tolerance model and parameters
        actual_failed_rate = number_of_failed_edges / total_edge_nums
        all_edges_run_completed = True if len(running_edges_list) <= 0 else False
        if all_edges_run_completed:
            status_to_report = None
            if enable_fault_tolerance:
                if actual_failed_rate >= fault_tolerance_rate:
                    status_to_report = ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED
                    self.report_exception_status(
                        running_edges_list, run_id=run_id, status=status_to_report)
                    return status_to_report
                else:
                    if number_of_killed_edges == total_edge_nums:
                        status_to_report = ServerConstants.MSG_MLOPS_SERVER_STATUS_KILLED
                    else:
                        status_to_report = ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED
            else:
                if number_of_failed_edges > 0:
                    status_to_report = ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED
                elif number_of_finished_edges == total_edge_nums:
                    status_to_report = ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED
                elif number_of_killed_edges == total_edge_nums:
                    status_to_report = ServerConstants.MSG_MLOPS_SERVER_STATUS_KILLED

            return status_to_report

    def parse_fault_tolerance_params(self, run_id):
        run_json = self.running_scheduler_contract.get(str(run_id), None)
        if run_json is None:
            return False, 0
        run_config = run_json.get("run_config", {})
        run_params = run_config.get("parameters", {})
        common_args = run_params.get("common_args", {})
        enable_fault_tolerance = common_args.get("enable_fault_tolerance", False)
        fault_tolerance_rate = common_args.get("fault_tolerance_rate", 0)
        return enable_fault_tolerance, fault_tolerance_rate

    def report_server_status(self, run_id, edge_id, server_id, status):
        self.status_reporter.report_server_id_status(
            run_id, status, edge_id=edge_id, server_id=server_id, server_agent_id=server_id, update_db=False)

    def report_exception_status(self, status):
        self.message_reporter.report_job_status(self.run_id, status)

    def status_center_process_slave_status_to_master_in_slave_agent(self, topic, payload):
        # Forward the status message to the sender queue of message center.
        self.message_center.send_message(topic, payload)

    def status_center_process_slave_status_to_mlops_in_slave_agent(self, topic, payload):
        # Forward the status message to message center.
        self.message_center.send_message(topic, payload)

    def status_center_request_job_status_from_master_in_slave_agent(self, topic, payload):
        # Parse the parameters
        payload_json = json.loads(payload)
        run_id = payload_json.get("run_id", None)
        master_id = payload_json.get("master_id", None)
        edge_id = payload_json.get("edge_id", None)

        # Request the job status from master agent.
        topic_request_job_status = f"{GeneralConstants.MSG_TOPIC_REQUEST_JOB_STATUS_PREFIX}{master_id}"
        payload_request_job_status = {"run_id": run_id, "edge_id": edge_id}
        self.message_center.send_message(topic_request_job_status, json.dumps(payload_request_job_status))

    def report_deployment_status(self, run_id, status):
        deployment_status_topic = "model_ops/model_device/return_deployment_status"
        deployment_status_payload = {"end_point_id": run_id, "end_point_name": "",
                                     "model_name": "",
                                     "model_url": "",
                                     "model_status": status,
                                     "timestamp": int(format(time.time_ns() / 1000.0, '.0f'))}
        logging.info(f"[StatusCenter] deployment_status_payload is sent to mlops: {deployment_status_payload}")

        self.message_center.send_message_json(deployment_status_topic, json.dumps(deployment_status_payload))
