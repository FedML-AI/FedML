import json
import logging
import os
import platform
import time
import traceback
import uuid
from os.path import expanduser

import multiprocessing
import psutil
import setproctitle

import fedml
from fedml.computing.scheduler.comm_utils import sys_utils
from .device_info_report_protocol import FedMLDeviceInfoReportProtocol
from .mlops_utils import MLOpsUtils
from .system_stats import SysStats
from ...computing.scheduler.comm_utils.job_monitor import JobMonitor
from ...computing.scheduler.scheduler_core.general_constants import GeneralConstants
from ...core.distributed.communication.mqtt.mqtt_manager import MqttManager


ROLE_DEVICE_JOB_TOTAL_MONITOR = 0
ROLE_DEVICE_INFO_REPORTER = 1
ROLE_ENDPOINT_MASTER = 2
ROLE_ENDPOINT_SLAVE = 3
ROLE_RUN_MASTER = 4
ROLE_RUN_SLAVE = 5
ROLE_ENDPOINT_LOGS = 6
ROLE_AUTO_SCALER = 7
ROLE_ENDPOINT_REPLICA_NUM = 8
ROLE_ENDPOINT_REPLICA_PERF = 9

ROLE_DEVICE_JOB_TOTAL_MONITOR_STR = "device_job_total"
ROLE_DEVICE_INFO_REPORTER_STR = "device_info"
ROLE_ENDPOINT_MASTER_STR = "endpoint_master"
ROLE_ENDPOINT_SLAVE_STR = "endpoint_slave"
ROLE_RUN_MASTER_STR = "run_master"
ROLE_RUN_SLAVE_STR = "run_slave"
ROLE_ENDPOINT_LOGS_STR = "endpoint_logs"
ROLE_AUTO_SCALER_STR = "autoscaler"
ROLE_ENDPOINT_REPLICA_NUM_STR = "endpoint_replica_num"
ROLE_ENDPOINT_REPLICA_PERF_STR = "endpoint_replica_perf"


class MLOpsDevicePerfStats(object):
    def __init__(self):
        self.device_realtime_stats_process = None
        self.device_realtime_stats_event = None
        self.monitor_run_slave_process = None
        self.monitor_run_master_process = None
        self.monitor_endpoint_master_process = None
        self.monitor_endpoint_slave_process = None
        self.monitor_endpoint_logs_process = None
        self.monitor_auto_scaler_process = None
        self.monitor_replica_num_process = None
        self.monitor_replica_perf_process = None
        self.job_total_monitor_process = None
        self.enable_job_total_monitor = False
        self.args = None
        self.device_id = None
        self.run_id = None
        self.edge_id = None
        self.is_client = True

    def report_device_realtime_stats(self, sys_args):
        self.setup_realtime_stats_process(sys_args)

    def stop_device_realtime_stats(self):
        if self.device_realtime_stats_event is not None:
            self.device_realtime_stats_event.set()

    def should_stop_device_realtime_stats(self):
        if self.device_realtime_stats_event is not None and self.device_realtime_stats_event.is_set():
            return True

        return False

    def setup_realtime_stats_process(self, sys_args):
        perf_stats = MLOpsDevicePerfStats()
        perf_stats.args = sys_args
        perf_stats.edge_id = getattr(sys_args, "edge_id", None)
        perf_stats.edge_id = getattr(sys_args, "client_id", None) if perf_stats.edge_id is None else perf_stats.edge_id
        perf_stats.edge_id = 0 if perf_stats.edge_id is None else perf_stats.edge_id
        perf_stats.device_id = getattr(sys_args, "device_id", 0)
        perf_stats.run_id = getattr(sys_args, "run_id", 0)
        perf_stats.is_client = self.is_client
        if self.device_realtime_stats_event is None:
            self.device_realtime_stats_event = multiprocessing.Event()
        self.device_realtime_stats_event.clear()
        perf_stats.device_realtime_stats_event = self.device_realtime_stats_event

        if platform.system() == "Windows":
            self.device_realtime_stats_process = multiprocessing.Process(
                target=perf_stats.report_device_realtime_stats_entry,
                args=(self.device_realtime_stats_event, ROLE_DEVICE_INFO_REPORTER, self.is_client,
                      GeneralConstants.get_monitor_process_name(
                          ROLE_DEVICE_INFO_REPORTER_STR, perf_stats.run_id, perf_stats.edge_id)))
        else:
            self.device_realtime_stats_process = fedml.get_process(
                target=perf_stats.report_device_realtime_stats_entry,
                args=(self.device_realtime_stats_event, ROLE_DEVICE_INFO_REPORTER, self.is_client,
                      GeneralConstants.get_monitor_process_name(
                          ROLE_DEVICE_INFO_REPORTER_STR, perf_stats.run_id, perf_stats.edge_id)))
        self.device_realtime_stats_process.start()

        if self.enable_job_total_monitor:
            if platform.system() == "Windows":
                self.job_total_monitor_process = multiprocessing.Process(
                    target=perf_stats.report_device_realtime_stats_entry,
                    args=(self.device_realtime_stats_event, ROLE_DEVICE_JOB_TOTAL_MONITOR, self.is_client,
                          GeneralConstants.get_monitor_process_name(
                              ROLE_DEVICE_JOB_TOTAL_MONITOR_STR, perf_stats.run_id, perf_stats.edge_id)))
            else:
                self.job_total_monitor_process = fedml.get_process(
                    target=perf_stats.report_device_realtime_stats_entry,
                    args=(self.device_realtime_stats_event, ROLE_DEVICE_JOB_TOTAL_MONITOR, self.is_client,
                          GeneralConstants.get_monitor_process_name(
                              ROLE_DEVICE_JOB_TOTAL_MONITOR_STR, perf_stats.run_id, perf_stats.edge_id)))
            self.job_total_monitor_process.start()
        else:
            if self.is_client:
                # Register endpoint master process
                if platform.system() == "Windows":
                    self.monitor_endpoint_master_process = multiprocessing.Process(
                        target=perf_stats.report_device_realtime_stats_entry,
                        args=(self.device_realtime_stats_event, ROLE_ENDPOINT_MASTER, True,
                              GeneralConstants.get_monitor_process_name(
                                  ROLE_ENDPOINT_MASTER_STR, perf_stats.run_id, perf_stats.edge_id)))
                else:
                    self.monitor_endpoint_master_process = fedml.get_process(
                        target=perf_stats.report_device_realtime_stats_entry,
                        args=(self.device_realtime_stats_event, ROLE_ENDPOINT_MASTER, True,
                              GeneralConstants.get_monitor_process_name(
                                  ROLE_ENDPOINT_MASTER_STR, perf_stats.run_id, perf_stats.edge_id)))
                self.monitor_endpoint_master_process.start()

                # Register endpoint slave process
                if platform.system() == "Windows":
                    self.monitor_endpoint_slave_process = multiprocessing.Process(
                        target=perf_stats.report_device_realtime_stats_entry,
                        args=(self.device_realtime_stats_event, ROLE_ENDPOINT_SLAVE, True,
                              GeneralConstants.get_monitor_process_name(
                                  ROLE_ENDPOINT_SLAVE_STR, perf_stats.run_id, perf_stats.edge_id)))
                else:
                    self.monitor_endpoint_slave_process = fedml.get_process(
                        target=perf_stats.report_device_realtime_stats_entry,
                        args=(self.device_realtime_stats_event, ROLE_ENDPOINT_SLAVE, True,
                              GeneralConstants.get_monitor_process_name(
                                  ROLE_ENDPOINT_SLAVE_STR, perf_stats.run_id, perf_stats.edge_id)))
                self.monitor_endpoint_slave_process.start()

                # Register run slave process
                if platform.system() == "Windows":
                    self.monitor_run_slave_process = multiprocessing.Process(
                        target=perf_stats.report_device_realtime_stats_entry,
                        args=(self.device_realtime_stats_event, ROLE_RUN_SLAVE, True,
                              GeneralConstants.get_monitor_process_name(
                                  ROLE_RUN_SLAVE_STR, perf_stats.run_id, perf_stats.edge_id)))
                else:
                    self.monitor_run_slave_process = fedml.get_process(
                        target=perf_stats.report_device_realtime_stats_entry,
                        args=(self.device_realtime_stats_event, ROLE_RUN_SLAVE, True,
                              GeneralConstants.get_monitor_process_name(
                                  ROLE_RUN_SLAVE_STR, perf_stats.run_id, perf_stats.edge_id)))
                self.monitor_run_slave_process.start()

                # Register endpoint logs process
                if platform.system() == "Windows":
                    self.monitor_endpoint_logs_process = multiprocessing.Process(
                        target=perf_stats.report_device_realtime_stats_entry,
                        args=(self.device_realtime_stats_event, ROLE_ENDPOINT_LOGS, True,
                              GeneralConstants.get_monitor_process_name(
                                  ROLE_ENDPOINT_LOGS_STR, perf_stats.run_id, perf_stats.edge_id)))
                else:
                    self.monitor_endpoint_logs_process = fedml.get_process(
                        target=perf_stats.report_device_realtime_stats_entry,
                        args=(self.device_realtime_stats_event, ROLE_ENDPOINT_LOGS, True,
                              GeneralConstants.get_monitor_process_name(
                                  ROLE_ENDPOINT_LOGS_STR, perf_stats.run_id, perf_stats.edge_id)))
                self.monitor_endpoint_logs_process.start()

                # Register auto-scaler process
                if platform.system() == "Windows":
                    self.monitor_auto_scaler_process = multiprocessing.Process(
                        target=perf_stats.report_device_realtime_stats_entry,
                        args=(self.device_realtime_stats_event, ROLE_AUTO_SCALER, True,
                              GeneralConstants.get_monitor_process_name(
                                  ROLE_AUTO_SCALER_STR, perf_stats.run_id, perf_stats.edge_id)))
                else:
                    self.monitor_auto_scaler_process = fedml.get_process(
                        target=perf_stats.report_device_realtime_stats_entry,
                        args=(self.device_realtime_stats_event, ROLE_AUTO_SCALER, True,
                              GeneralConstants.get_monitor_process_name(
                                  ROLE_AUTO_SCALER_STR, perf_stats.run_id, perf_stats.edge_id)))
                self.monitor_auto_scaler_process.start()

                # Register replica number report channel
                if platform.system() == "Windows":
                    self.monitor_replica_num_process = multiprocessing.Process(
                        target=perf_stats.report_device_realtime_stats_entry,
                        args=(self.device_realtime_stats_event, ROLE_ENDPOINT_REPLICA_NUM, True,
                              GeneralConstants.get_monitor_process_name(
                                  ROLE_ENDPOINT_REPLICA_NUM_STR, perf_stats.run_id, perf_stats.edge_id)))
                else:
                    self.monitor_replica_num_process = fedml.get_process(
                        target=perf_stats.report_device_realtime_stats_entry,
                        args=(self.device_realtime_stats_event, ROLE_ENDPOINT_REPLICA_NUM, True,
                              GeneralConstants.get_monitor_process_name(
                                  ROLE_ENDPOINT_REPLICA_NUM_STR, perf_stats.run_id, perf_stats.edge_id)))
                self.monitor_replica_num_process.start()

                # Register replica performance report channel
                if platform.system() == "Windows":
                    self.monitor_replica_perf_process = multiprocessing.Process(
                        target=perf_stats.report_device_realtime_stats_entry,
                        args=(self.device_realtime_stats_event, ROLE_ENDPOINT_REPLICA_PERF, True,
                              GeneralConstants.get_monitor_process_name(
                                  ROLE_ENDPOINT_REPLICA_PERF_STR, perf_stats.run_id, perf_stats.edge_id)))

                else:
                    self.monitor_replica_perf_process = fedml.get_process(
                        target=perf_stats.report_device_realtime_stats_entry,
                        args=(self.device_realtime_stats_event, ROLE_ENDPOINT_REPLICA_PERF, True,
                              GeneralConstants.get_monitor_process_name(
                                  ROLE_ENDPOINT_REPLICA_PERF_STR, perf_stats.run_id, perf_stats.edge_id)))
                self.monitor_replica_perf_process.start()
            else:
                if platform.system() == "Windows":
                    self.monitor_run_master_process = multiprocessing.Process(
                        target=perf_stats.report_device_realtime_stats_entry,
                        args=(self.device_realtime_stats_event, ROLE_RUN_MASTER, False,
                              GeneralConstants.get_monitor_process_name(
                                  ROLE_RUN_MASTER_STR, perf_stats.run_id, perf_stats.edge_id)))
                else:
                    self.monitor_run_master_process = fedml.get_process(
                        target=perf_stats.report_device_realtime_stats_entry,
                        args=(self.device_realtime_stats_event, ROLE_RUN_MASTER, False,
                              GeneralConstants.get_monitor_process_name(
                                  ROLE_RUN_MASTER_STR, perf_stats.run_id, perf_stats.edge_id)))
                self.monitor_run_master_process.start()

    def report_device_realtime_stats_entry(self, sys_event, role, is_client=False, process_name=None):
        if process_name is not None:
            setproctitle.setproctitle(process_name)

        # print(f"Report device realtime stats, process id {os.getpid()}, name {process_name}")

        self.device_realtime_stats_event = sys_event
        mqtt_mgr = MqttManager(
            self.args.mqtt_config_path["BROKER_HOST"],
            self.args.mqtt_config_path["BROKER_PORT"],
            self.args.mqtt_config_path["MQTT_USER"],
            self.args.mqtt_config_path["MQTT_PWD"],
            180,
            "FedML_Metrics_DevicePerf_{}_{}_{}".format(str(self.args.device_id), str(self.edge_id), str(uuid.uuid4()))
        )
        mqtt_mgr.connect()
        mqtt_mgr.loop_start()

        parent_pid = psutil.Process(os.getpid()).ppid()
        sys_stats_obj = SysStats(process_id=parent_pid)

        device_info_reporter = FedMLDeviceInfoReportProtocol(run_id=self.run_id, mqtt_mgr=mqtt_mgr)

        JobMonitor.get_instance().mqtt_config = self.args.mqtt_config_path

        # Notify MLOps with system information.
        sleep_time_interval = 10
        time_interval_map = {
            ROLE_DEVICE_INFO_REPORTER: 10, ROLE_RUN_SLAVE: 60, ROLE_RUN_MASTER: 70,
            ROLE_ENDPOINT_SLAVE: 80, ROLE_ENDPOINT_MASTER: 90, ROLE_ENDPOINT_LOGS: 30,
            ROLE_AUTO_SCALER: 60, ROLE_ENDPOINT_REPLICA_NUM: 30, ROLE_ENDPOINT_REPLICA_PERF: 30
        }

        job_monitor_obj = None
        if role == ROLE_AUTO_SCALER or role == ROLE_DEVICE_JOB_TOTAL_MONITOR:
            # job_monitor Should be initialized once
            job_monitor_obj = JobMonitor.get_instance()

        sleep_time_interval_for_device_info = 60
        sleep_time_interval_for_client_monitor = 30
        sleep_time_interval_for_server_monitor = 60

        while not self.should_stop_device_realtime_stats():
            if self.enable_job_total_monitor:
                if role == ROLE_DEVICE_INFO_REPORTER:
                    time.sleep(sleep_time_interval_for_device_info)
                elif role == ROLE_DEVICE_JOB_TOTAL_MONITOR:
                    time.sleep(sleep_time_interval_for_client_monitor if is_client
                               else sleep_time_interval_for_server_monitor)
            else:
                time.sleep(time_interval_map[role])

            try:
                if role == ROLE_DEVICE_INFO_REPORTER:
                    MLOpsDevicePerfStats.report_gpu_device_info(self.edge_id, mqtt_mgr=mqtt_mgr)
                elif role == ROLE_RUN_SLAVE:
                    JobMonitor.get_instance().monitor_slave_run_process_status()
                elif role == ROLE_RUN_MASTER:
                    JobMonitor.get_instance().monitor_master_run_process_status(
                        self.edge_id, device_info_reporter=device_info_reporter)
                elif role == ROLE_ENDPOINT_SLAVE:
                    JobMonitor.get_instance().monitor_slave_endpoint_status()
                elif role == ROLE_ENDPOINT_MASTER:
                    JobMonitor.get_instance().monitor_master_endpoint_status()
                elif role == ROLE_ENDPOINT_LOGS:
                    JobMonitor.get_instance().monitor_endpoint_logs()
                elif role == ROLE_ENDPOINT_REPLICA_NUM:
                    JobMonitor.get_instance().monitor_replicas_number()
                elif role == ROLE_ENDPOINT_REPLICA_PERF:
                    JobMonitor.get_instance().monitor_replicas_perf(self.edge_id, mqtt_mgr=mqtt_mgr)
                elif role == ROLE_AUTO_SCALER:
                    job_monitor_obj.autoscaler_reconcile_after_interval()
                elif role == ROLE_DEVICE_JOB_TOTAL_MONITOR:
                    if is_client:
                        JobMonitor.get_instance().monitor_slave_run_process_status()
                        JobMonitor.get_instance().monitor_slave_endpoint_status()
                        JobMonitor.get_instance().monitor_master_endpoint_status()
                        JobMonitor.get_instance().monitor_endpoint_logs()
                        JobMonitor.get_instance().monitor_replicas_number()
                        JobMonitor.get_instance().monitor_replicas_perf(self.edge_id, mqtt_mgr=mqtt_mgr)
                        job_monitor_obj.autoscaler_reconcile_after_interval()
                    else:
                        JobMonitor.get_instance().monitor_master_run_process_status(
                            self.edge_id, device_info_reporter=device_info_reporter)

            except Exception as e:
                logging.error(f"exception {e} when reporting device pref: {traceback.format_exc()}.")
                pass

            if role == ROLE_DEVICE_INFO_REPORTER:
                self.check_fedml_client_parent_process()

                self.check_fedml_server_parent_process()

        logging.info("Device metrics process is about to exit.")
        mqtt_mgr.loop_stop()
        mqtt_mgr.disconnect()

    @staticmethod
    def report_gpu_device_info(edge_id, mqtt_mgr=None):
        total_mem, free_mem, total_disk_size, free_disk_size, cup_utilization, cpu_cores, gpu_cores_total, \
            gpu_cores_available, sent_bytes, recv_bytes, gpu_available_ids = sys_utils.get_sys_realtime_stats()

        topic_name = "ml_client/mlops/gpu_device_info"

        # We should report realtime available gpu count to MLOps, not from local redis cache.
        # Use gpu_available_ids from sys_utils.get_sys_realtime_stats()
        # Do not use the following two lines as the realtime available gpu ids.
        # gpu_available_ids = JobRunnerUtils.get_available_gpu_id_list(edge_id)
        # gpu_available_ids = JobRunnerUtils.trim_unavailable_gpu_ids(gpu_available_ids)
        gpu_cores_available = len(gpu_available_ids)
        deploy_worker_id_list = list()
        try:
            deploy_worker_id_list = json.loads(os.environ.get("FEDML_DEPLOY_WORKER_IDS", "[]"))
        except Exception as e:
            logging.error(f"Exception {e} occurred when parsing FEDML_DEPLOY_WORKER_IDS. "
                          f"Traceback: {traceback.format_exc()}.")
            pass
        device_info_json = {
            "edgeId": edge_id,
            "deployMasterId": os.environ.get("FEDML_DEPLOY_MASTER_ID", ""),
            "deployWorkerIds": deploy_worker_id_list,
            "memoryTotal": round(total_mem * MLOpsUtils.BYTES_TO_GB, 2),
            "memoryAvailable": round(free_mem * MLOpsUtils.BYTES_TO_GB, 2),
            "diskSpaceTotal": round(total_disk_size * MLOpsUtils.BYTES_TO_GB, 2),
            "diskSpaceAvailable": round(free_disk_size * MLOpsUtils.BYTES_TO_GB, 2),
            "cpuUtilization": round(cup_utilization, 2),
            "cpuCores": cpu_cores,
            "gpuCoresTotal": gpu_cores_total,
            "gpuCoresAvailable": gpu_cores_available,
            "gpuAvailableIds": gpu_available_ids,
            "networkTraffic": sent_bytes + recv_bytes,
            "updateTime": int(MLOpsUtils.get_ntp_time())
        }
        message_json = json.dumps(device_info_json)
        if mqtt_mgr is not None:
            mqtt_mgr.send_message_json(topic_name, message_json)

    def check_fedml_client_parent_process(self):
        if not self.is_client:
            return

        # inspection PyBroadException
        try:
            home_dir = expanduser("~")
            fedml_ppids_dir = os.path.join(home_dir, ".fedml", "fedml-client", "fedml", "data", "ppids")
            if not os.path.exists(fedml_ppids_dir):
                return

            should_logout = True
            file_list = os.listdir(fedml_ppids_dir)
            if len(file_list) <= 0:
                should_logout = False
            else:
                for parent_pid in file_list:
                    if not psutil.pid_exists(int(parent_pid)):
                        os.remove(os.path.join(fedml_ppids_dir, parent_pid))
                    else:
                        should_logout = False

            if should_logout:
                print(f"Parent client process {file_list} has been killed, so fedml will exit.")
                logging.info(f"Parent client process {file_list} has been killed, so fedml will exit.")
                os.system("fedml logout")
        except Exception:
            pass

    def check_fedml_server_parent_process(self):
        if self.is_client:
            return

        # inspection PyBroadException
        try:
            home_dir = expanduser("~")
            fedml_ppids_dir = os.path.join(home_dir, ".fedml", "fedml-server", "fedml", "data", "ppids")
            if not os.path.exists(fedml_ppids_dir):
                return

            should_logout = True
            file_list = os.listdir(fedml_ppids_dir)
            if len(file_list) <= 0:
                should_logout = False
            else:
                for parent_pid in file_list:
                    if not psutil.pid_exists(int(parent_pid)):
                        os.remove(os.path.join(fedml_ppids_dir, parent_pid))
                    else:
                        should_logout = False

            if should_logout:
                print(f"Parent server process {file_list} has been killed, so fedml will exit.")
                logging.info(f"Parent server process {file_list} has been killed, so fedml will exit.")
                os.system("fedml logout -s")
        except Exception:
            pass
