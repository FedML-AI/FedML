import json
import logging
import os
import platform
import time
import traceback
import uuid

import multiprocess as multiprocessing
import psutil
import setproctitle

import fedml
from .mlops_utils import MLOpsUtils
from .system_stats import SysStats
from ...computing.scheduler.scheduler_core.general_constants import GeneralConstants
from ...core.distributed.communication.mqtt.mqtt_manager import MqttManager


class MLOpsJobPerfStats(object):
    JOB_PERF_PROCESS_TAG = "job_perf"

    def __init__(self):
        self.job_stats_process = None
        self.job_stats_event = None
        self.args = None
        self.device_id = None
        self.run_id = None
        self.edge_id = None
        self.job_process_id_map = dict()
        self.job_stats_obj_map = dict()

    def add_job(self, job_id, process_id):
        self.job_process_id_map[job_id] = process_id

    @staticmethod
    def report_system_metric(run_id, edge_id, metric_json=None,
                             mqtt_mgr=None, sys_stats_obj=None):
        # if not self.comm_sanity_check():
        #     return
        if run_id is None:
            return
        run_id_str = str(run_id).strip()
        if run_id_str == "0" or run_id_str == "":
            return

        topic_name = "fl_client/mlops/system_performance"
        if metric_json is None:
            if sys_stats_obj is None:
                sys_stats_obj = SysStats(process_id=os.getpid())
            sys_stats_obj.produce_info()

            current_time_ms = MLOpsUtils.get_ntp_time()
            if current_time_ms is None:
                current_time = int(time.time() * 1000)
            else:
                current_time = int(current_time_ms)

            metric_json = {
                "run_id": run_id,
                "edge_id": edge_id,
                "cpu_utilization": round(
                    sys_stats_obj.get_cpu_utilization(), 4
                ),
                "SystemMemoryUtilization": round(
                    sys_stats_obj.get_system_memory_utilization(), 4
                ),
                "process_memory_in_use": round(
                    sys_stats_obj.get_process_memory_in_use(), 4
                ),
                "process_memory_in_use_size": round(
                    sys_stats_obj.get_process_memory_in_use_size(), 4
                ),
                "process_memory_available": round(
                    sys_stats_obj.get_process_memory_available(), 4
                ),
                "process_cpu_threads_in_use": round(
                    sys_stats_obj.get_process_cpu_threads_in_use(), 4
                ),
                "disk_utilization": round(
                    sys_stats_obj.get_disk_utilization(), 4
                ),
                "network_traffic": round(
                    sys_stats_obj.get_network_traffic(), 4
                ),
                "gpu_utilization": round(
                    sys_stats_obj.get_gpu_utilization(), 4
                ),
                "gpu_temp": round(sys_stats_obj.get_gpu_temp(), 4),
                "gpu_time_spent_accessing_memory": round(
                    sys_stats_obj.get_gpu_time_spent_accessing_memory(), 4
                ),
                "gpu_memory_allocated": round(
                    sys_stats_obj.get_gpu_memory_allocated(), 4
                ),
                "gpu_power_usage": round(
                    sys_stats_obj.get_gpu_power_usage(), 4
                ),
                "timestamp": int(current_time)
            }

            gpu_metrics_list = list()
            if sys_stats_obj.metrics_of_all_gpus is not None:
                for gpu_metric_item in sys_stats_obj.metrics_of_all_gpus:
                    gpu_metric_dict = {
                        "gpu_id": gpu_metric_item.gpu_id,
                        "gpu_name": gpu_metric_item.gpu_name,
                        "gpu_utilization": gpu_metric_item.gpu_utilization,
                        "gpu_memory_allocated": gpu_metric_item.gpu_memory_allocated,
                        "gpu_temp": gpu_metric_item.gpu_temp,
                        "gpu_power_usage": gpu_metric_item.gpu_power_usage,
                        "gpu_time_spent_accessing_memory": gpu_metric_item.gpu_time_spent_accessing_memory
                    }
                    gpu_metrics_list.append(gpu_metric_dict)
            metric_json["metrics_of_all_gpus"] = gpu_metrics_list

        message_json = json.dumps(metric_json)
        if mqtt_mgr is not None:
            mqtt_mgr.send_message_json(topic_name, message_json)

    def stop_job_stats(self):
        if self.job_stats_event is not None:
            self.job_stats_event.set()

    def should_stop_job_stats(self):
        if self.job_stats_event is not None and self.job_stats_event.is_set():
            return True

        return False

    def setup_job_stats_process(self, sys_args):
        if self.job_stats_process is not None and psutil.pid_exists(self.job_stats_process.pid):
            return

        perf_stats = MLOpsJobPerfStats()
        perf_stats.args = sys_args
        perf_stats.edge_id = getattr(sys_args, "edge_id", None)
        perf_stats.edge_id = getattr(sys_args, "client_id", None) if perf_stats.edge_id is None else perf_stats.edge_id
        perf_stats.edge_id = 0 if perf_stats.edge_id is None else perf_stats.edge_id
        perf_stats.device_id = getattr(sys_args, "device_id", 0)
        perf_stats.run_id = getattr(sys_args, "run_id", 0)
        if self.job_stats_event is None:
            self.job_stats_event = multiprocessing.Event()
        self.job_stats_event.clear()
        perf_stats.job_stats_event = self.job_stats_event
        perf_stats.job_process_id_map = self.job_process_id_map
        if platform.system() == "Windows":
            self.job_stats_process = multiprocessing.Process(
                target=perf_stats.report_job_stats_entry,
                args=(self.job_stats_event, GeneralConstants.get_monitor_process_name(
                    MLOpsJobPerfStats.JOB_PERF_PROCESS_TAG, perf_stats.run_id, perf_stats.edge_id)))
        else:
            self.job_stats_process = fedml.get_process(
                target=perf_stats.report_job_stats_entry,
                args=(self.job_stats_event, GeneralConstants.get_monitor_process_name(
                    MLOpsJobPerfStats.JOB_PERF_PROCESS_TAG, perf_stats.run_id, perf_stats.edge_id)))
        self.job_stats_process.start()

    def report_job_stats(self, sys_args):
        self.setup_job_stats_process(sys_args)

    def report_job_stats_entry(self, sys_event, process_name):
        if process_name is not None:
            setproctitle.setproctitle(process_name)

        # print(f"Report job realtime stats, process id {os.getpid()}, name {process_name}")

        self.job_stats_event = sys_event
        mqtt_mgr = MqttManager(
            self.args.mqtt_config_path["BROKER_HOST"],
            self.args.mqtt_config_path["BROKER_PORT"],
            self.args.mqtt_config_path["MQTT_USER"],
            self.args.mqtt_config_path["MQTT_PWD"],
            180,
            "FedML_Metrics_JobPerf_{}_{}_{}".format(str(self.device_id), str(self.edge_id), str(uuid.uuid4()))
        )
        mqtt_mgr.connect()
        mqtt_mgr.loop_start()

        # Notify MLOps with system information.
        while not self.should_stop_job_stats():
            for job_id, process_id in self.job_process_id_map.items():
                try:
                    if self.job_stats_obj_map.get(job_id, None) is None:
                        self.job_stats_obj_map[job_id] = SysStats(process_id=process_id)

                    MLOpsJobPerfStats.report_system_metric(job_id, self.edge_id,
                                                           mqtt_mgr=mqtt_mgr,
                                                           sys_stats_obj=self.job_stats_obj_map[job_id])
                except Exception as e:
                    logging.debug("exception when reporting job pref: {}.".format(traceback.format_exc()))
                    pass

            time.sleep(10)

        logging.info("Job metrics process is about to exit.")
        mqtt_mgr.loop_stop()
        mqtt_mgr.disconnect()
        self.job_stats_process = None
