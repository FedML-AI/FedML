import json
import logging
import multiprocessing
import platform
import queue
import os
import time
import traceback

import setproctitle

from ..scheduler_entry.constants import Constants
from ....core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from ..master.server_constants import ServerConstants
from ....core.mlops.mlops_runtime_log_daemon import MLOpsRuntimeLogDaemon
from ..comm_utils import sys_utils
from .server_data_interface import FedMLServerDataInterface
from ....core.mlops.mlops_utils import MLOpsUtils
from ..scheduler_core.log_manager import LogsManager
from ..scheduler_core.metrics_manager import MetricsManager
from fedml.utils.debugging import debug
from ..scheduler_core.status_center import JobStatus
from ..scheduler_core.compute_cache_manager import ComputeCacheManager
from ..scheduler_core.general_constants import GeneralConstants
from ..scheduler_core.scheduler_base_job_runner import FedMLSchedulerBaseJobRunner, RunnerError, RunnerCompletedError
from abc import ABC, abstractmethod
from ..scheduler_core.scheduler_matcher import SchedulerMatcher
import fedml


class FedMLBaseMasterJobRunner(FedMLSchedulerBaseJobRunner, ABC):
    debug_cloud_server = False

    def __init__(self, args, run_id=0, request_json=None, agent_config=None, edge_id=0,
                 cuda_visible_gpu_ids_str=None,
                 agent_data_dir=None, agent_package_download_dir=None,
                 agent_package_unzip_dir=None, agent_log_file_dir=None):
        FedMLSchedulerBaseJobRunner.__init__(
            self, args, edge_id=edge_id, request_json=request_json, agent_config=agent_config, run_id=run_id,
            cuda_visible_gpu_ids_str=cuda_visible_gpu_ids_str, agent_data_dir=agent_data_dir,
            agent_package_download_dir=agent_package_download_dir,
            agent_package_unzip_dir=agent_package_unzip_dir,
            agent_log_file_dir=agent_package_unzip_dir,
            is_master_runner=True
        )

        self.run_edge_id_status_queue = multiprocessing.Manager().Queue()
        self.run_metrics_queue = multiprocessing.Manager().Queue()
        self.run_events_queue = multiprocessing.Manager().Queue()
        self.run_artifacts_queue = multiprocessing.Manager().Queue()
        self.run_logs_queue = multiprocessing.Manager().Queue()
        self.run_edge_device_info_queue = multiprocessing.Manager().Queue()
        self.run_edge_device_info_global_queue = multiprocessing.Manager().Queue()
        self.run_extend_queue_list = None
        self.async_check_timeout = 0
        self.enable_async_cluster = False
        self.origin_fedml_config_object = None
        self.server_agent_id = 0
        if request_json is not None:
            self.server_agent_id = request_json.get("server_id", 0)
        self.fedml_data_base_package_dir = os.path.join("/", "fedml", "data")
        self.fedml_data_local_package_dir = os.path.join("/", "fedml", "fedml-package", "fedml", "data")
        self.fedml_data_dir = self.fedml_data_base_package_dir
        self.fedml_config_dir = os.path.join("/", "fedml", "conf")

    @debug
    def run(
            self, process_event, completed_event, edge_id_status_queue=None,
            edge_device_info_queue=None, run_metrics_queue=None, run_event_queue=None,
            run_artifacts_queue=None, run_logs_queue=None, edge_device_info_global_queue=None,
            run_extend_queue_list=None, sender_message_center_queue=None, listener_message_queue=None,
            status_center_queue=None, process_name=None
    ):
        if process_name is not None:
            setproctitle.setproctitle(process_name)

        print(f"Master job runner process id {os.getpid()}, name {process_name}, run id {self.run_id}")

        if platform.system() != "Windows":
            os.setsid()

        os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
        os.environ.setdefault('PYTHONWARNINGS', 'ignore:semaphore_tracker:UserWarning')

        self.run_process_event = process_event
        self.run_process_completed_event = completed_event
        try:
            MLOpsUtils.set_ntp_offset(self.ntp_offset)

            self.rebuild_message_status_center(sender_message_center_queue, listener_message_queue, status_center_queue)

            self.run_impl(
                edge_id_status_queue, edge_device_info_queue, run_metrics_queue,
                run_event_queue, run_artifacts_queue, run_logs_queue, edge_device_info_global_queue,
                run_extend_queue_list=run_extend_queue_list, sender_message_queue=sender_message_center_queue,
                listener_message_queue=listener_message_queue, status_center_queue=status_center_queue
            )
        except RunnerError:
            logging.info("Runner stopped.")
            self.status_reporter.report_server_id_status(
                self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_KILLED, edge_id=self.edge_id,
                server_id=self.edge_id, server_agent_id=self.edge_id)
        except RunnerCompletedError:
            logging.info("Runner completed.")
        except Exception as e:
            logging.error("Runner exits with exceptions. {}".format(traceback.format_exc()))
            self.status_reporter.report_server_id_status(
                self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED, edge_id=self.edge_id,
                server_id=self.edge_id, server_agent_id=self.edge_id)
        finally:
            logging.info("Release resources.")
            self._process_run_metrics_queue(run_metrics_queue)
            self._process_run_logs_queue(run_logs_queue)
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(self.run_id, self.edge_id)
            if self.mlops_metrics is not None:
                self.mlops_metrics.stop_sys_perf()
            time.sleep(3)
            self.cleanup_runner_process(self.run_id)
            ServerConstants.cleanup_learning_process(self.run_id)
            ServerConstants.cleanup_bootstrap_process(self.run_id)

    def cleanup_runner_process(self, run_id):
        ServerConstants.cleanup_run_process(run_id)

    @debug
    @abstractmethod
    def run_impl(
            self, edge_id_status_queue, edge_device_info_queue, run_metrics_queue,
            run_event_queue, run_artifacts_queue, run_logs_queue, edge_device_info_global_queue,
            run_extend_queue_list=None, sender_message_queue=None, listener_message_queue=None,
            status_center_queue=None
    ):
        run_id = self.request_json["runId"]
        run_config = self.request_json["run_config"]
        data_config = run_config["data_config"]
        edge_ids = self.request_json["edgeids"]

        self.check_runner_stop_event()

        self.run_id = run_id
        self.args.run_id = self.run_id
        MLOpsRuntimeLog.get_instance(self.args).init_logs(log_level=logging.INFO)

        logging.info("Detect all status of Edge ids: " + str(edge_ids))

        self.status_reporter.report_server_id_status(
            self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_STARTING, edge_id=self.edge_id,
            server_id=self.edge_id, server_agent_id=self.edge_id)

        status_ok, active_edge_info_dict, inactivate_edges = self.detect_edges_status(
            edge_device_info_queue, edge_device_info_global_queue=edge_device_info_global_queue,
            callback_when_edges_ready=self.send_training_request_to_edges)
        logging.info(f"Status OK: {status_ok}, Active edge info dict: {active_edge_info_dict}, "
                     f"inactivate edges: {inactivate_edges}")
        if not status_ok:
            logging.error(f"Status of edge device is not OK. Active edge info dict: {active_edge_info_dict}, "
                          f"Inactivate edges: {inactivate_edges}")
            return

        if not self.should_continue_run_job(run_id):
            if FedMLBaseMasterJobRunner.debug_cloud_server:
                while True:
                    time.sleep(30)
            # Check if the run status is normal
            self.aggregate_run_metrics_logs(
                run_id, edge_ids, edge_id_status_queue, edge_device_info_queue,
                edge_device_info_global_queue,
                run_metrics_queue, run_logs_queue)
            return

        # Start the server job
        self.start_runner_process(
            run_id, self.request_json, edge_id=self.edge_id, is_server_job=True,
            sender_message_queue=sender_message_queue,
            listener_message_queue=listener_message_queue,
            status_center_queue=status_center_queue,
            process_name=GeneralConstants.get_launch_master_user_process_name(run_id, self.edge_id)
        )

        # Check if the run status is normal
        self.aggregate_run_metrics_logs(
            run_id, edge_ids, edge_id_status_queue, edge_device_info_queue,
            edge_device_info_global_queue,
            run_metrics_queue, run_logs_queue)

    @abstractmethod
    def _generate_extend_queue_list(self):
        return list()

    def aggregate_run_metrics_logs(
            self, run_id, edge_id_list, edge_id_status_queue, edge_device_info_queue,
            edge_device_info_global_queue, run_metrics_queue, run_logs_queue):

        ComputeCacheManager.get_instance().set_redis_params()

        while True:
            self.check_runner_stop_event()

            # Process run metrics
            self._process_run_metrics_queue(run_metrics_queue)

            # Process run logs
            self._process_run_logs_queue(run_logs_queue)

            # Check the job status
            job_status = ComputeCacheManager.get_instance().get_status_cache().get_job_status(run_id)
            if JobStatus.is_job_completed(job_status):
                break

    def _process_run_metrics_queue(self, run_metrics_queue):
        # Fetch metrics from the run metrics queue
        while True:
            try:
                metrics_item = run_metrics_queue.get(block=False, timeout=3)
                MetricsManager.get_instance().save_metrics(metrics_item)
                metric_json = json.loads(metrics_item)
                if metric_json.get("is_endpoint", False):
                    metric_json().pop("is_endpoint")
                    self.mlops_metrics.report_endpoint_metric({}, payload=json.dumps(metric_json))
                else:
                    self.mlops_metrics.report_server_training_metric({}, payload=metrics_item)
            except queue.Empty as e:  # If queue is empty, then break loop
                break

    def _process_run_logs_queue(self, run_logs_queue):
        # Fetch logs from the run logs queue
        while True:
            try:
                logs_item = run_logs_queue.get(block=False, timeout=3)
                LogsManager.save_logs(logs_item)
            except queue.Empty as e:  # If queue is empty, then break loop
                break

    def run_server_job(
            self, process_event, completed_event, edge_id_status_queue=None,
            edge_device_info_queue=None, run_metrics_queue=None, run_event_queue=None,
            run_artifacts_queue=None, run_logs_queue=None, edge_device_info_global_queue=None,
            run_extend_queue_list=None, sender_message_center_queue=None, listener_message_queue=None,
            status_center_queue=None, process_name=None
    ):
        if process_name is not None:
            setproctitle.setproctitle(process_name)

        print(f"Server runner process id {os.getpid()}, name {process_name}. run id {self.run_id}")

        if platform.system() != "Windows":
            os.setsid()

        os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
        os.environ.setdefault('PYTHONWARNINGS', 'ignore:semaphore_tracker:UserWarning')

        self.run_process_event = process_event
        self.run_process_completed_event = completed_event
        try:
            MLOpsUtils.set_ntp_offset(self.ntp_offset)

            self.rebuild_message_status_center(sender_message_center_queue, listener_message_queue, status_center_queue)

            self.run_server_job_impl(process_event, completed_event,
                                     message_center_queue=sender_message_center_queue)
        except RunnerError:
            logging.info("Runner stopped.")
            self.status_reporter.report_server_id_status(
                self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_KILLED, edge_id=self.edge_id,
                server_id=self.edge_id, server_agent_id=self.edge_id)
        except RunnerCompletedError:
            logging.info("Runner completed.")
        except Exception as e:
            logging.error("Runner exits with exceptions. {}".format(traceback.format_exc()))
            self.status_reporter.report_server_id_status(
                self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED, edge_id=self.edge_id,
                server_id=self.edge_id, server_agent_id=self.edge_id)
        finally:
            logging.info("Release resources.")
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(self.run_id, self.edge_id)
            if self.mlops_metrics is not None:
                self.mlops_metrics.stop_sys_perf()
            time.sleep(3)
            ServerConstants.cleanup_run_process(self.run_id)
            ServerConstants.cleanup_learning_process(self.run_id)
            ServerConstants.cleanup_bootstrap_process(self.run_id)

    def run_server_job_impl(self, process_event, completed_event,
                            message_center_queue=None):
        run_id = self.request_json["runId"]
        run_config = self.request_json["run_config"]
        data_config = run_config["data_config"]
        edge_ids = self.request_json["edgeids"]

        self.check_runner_stop_event()

        self.run_id = run_id
        self.args.run_id = self.run_id
        MLOpsRuntimeLog.get_instance(self.args).init_logs(log_level=logging.INFO)

        self.status_reporter.report_server_id_status(
            run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING, edge_id=self.edge_id,
            server_id=self.edge_id, server_agent_id=self.edge_id)

        # get training params
        private_local_data_dir = data_config.get("privateLocalData", "")
        is_using_local_data = 0
        # if private_local_data_dir is not None and len(str(private_local_data_dir).strip(' ')) > 0:
        #     is_using_local_data = 1

        # start a run according to the hyper-parameters
        # fedml_local_data_dir = self.cur_dir + "/fedml_data/run_" + run_id_str + "_edge_" + str(edge_id)
        fedml_local_data_dir = os.path.join(self.cur_dir, "fedml_data")
        fedml_local_config_dir = os.path.join(self.cur_dir, "fedml_config")
        if is_using_local_data:
            fedml_local_data_dir = private_local_data_dir
        self.fedml_data_dir = self.fedml_data_local_package_dir

        self.check_runner_stop_event()

        logging.info("download packages and run the bootstrap script...")

        # update local config with real time parameters from server and dynamically replace variables value
        unzip_package_path, fedml_config_object = self.update_local_fedml_config(run_id, run_config)
        if unzip_package_path is None or fedml_config_object is None:
            logging.info("failed to update local fedml config.")
            self.check_runner_stop_event()
            self.report_exception_status(run_id)
            return

        logging.info("cleanup the previous aggregation process and check downloaded packages...")

        entry_file_config = fedml_config_object["entry_config"]
        dynamic_args_config = fedml_config_object["dynamic_args"]
        entry_file = str(entry_file_config["entry_file"]).replace('\\', os.sep).replace('/', os.sep)
        entry_file = os.path.basename(entry_file)
        conf_file = entry_file_config["conf_file"]
        conf_file = str(conf_file).replace('\\', os.sep).replace('/', os.sep)
        ServerConstants.cleanup_learning_process(run_id)
        self.check_runner_stop_event()
        if not os.path.exists(unzip_package_path):
            logging.info("failed to unzip file.")
            self.check_runner_stop_event()
            self.report_exception_status(run_id)
            return
        os.chdir(os.path.join(unzip_package_path, "fedml"))

        self.check_runner_stop_event()

        logging.info("starting the server user process...")

        entry_file_full_path = os.path.join(unzip_package_path, "fedml", entry_file)
        conf_file_full_path = os.path.join(unzip_package_path, "fedml", conf_file)
        logging.info("                          ")
        logging.info("                          ")
        logging.info("====Your Run Logs Begin===")

        process, is_launch_task, error_list = self.execute_job_task(
            unzip_package_path=unzip_package_path, entry_file_full_path=entry_file_full_path,
            conf_file_full_path=conf_file_full_path, dynamic_args_config=dynamic_args_config,
            fedml_config_object=self.fedml_config_object)

        logging.info("====Your Run Logs End===")
        logging.info("                        ")
        logging.info("                        ")

        ret_code, out, err = process.returncode, None, None
        is_run_ok = sys_utils.is_runner_finished_normally(process.pid)
        if is_launch_task:
            is_run_ok = True
        if error_list is not None and len(error_list) > 0:
            is_run_ok = False
        if ret_code is None or ret_code <= 0:
            self.check_runner_stop_event()

            if is_run_ok:
                if out is not None:
                    out_str = sys_utils.decode_our_err_result(out)
                    if out_str != "":
                        logging.info("{}".format(out_str))

                self.status_reporter.report_server_id_status(
                    run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED, edge_id=self.edge_id,
                    server_id=self.edge_id, server_agent_id=self.edge_id)

                if is_launch_task:
                    sys_utils.log_return_info(f"job {run_id}", 0)
                else:
                    sys_utils.log_return_info(entry_file, 0)
        else:
            is_run_ok = False

        if not is_run_ok:
            # If the run status is killed or finished, then return with the normal state.
            current_job = FedMLServerDataInterface.get_instance().get_job_by_id(run_id)
            if current_job is not None and (current_job.status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED or
                                            current_job.status == ServerConstants.MSG_MLOPS_SERVER_STATUS_KILLED):
                return

            self.check_runner_stop_event()

            logging.error("failed to run the aggregation process...")

            if err is not None:
                err_str = sys_utils.decode_our_err_result(err)
                if err_str != "":
                    logging.error("{}".format(err_str))

            if is_launch_task:
                sys_utils.log_return_info(f"job {run_id}", ret_code)
            else:
                sys_utils.log_return_info(entry_file, ret_code)

            self.report_exception_status(run_id)

    @abstractmethod
    def _generate_job_runner_instance(self, args, run_id=None, request_json=None, agent_config=None, edge_id=None):
        return None

    def start_runner_process(
            self, run_id, request_json, edge_id=None, is_server_job=False,
            sender_message_queue=None, listener_message_queue=None,
            status_center_queue=None, process_name=None
    ):
        server_runner = self._generate_job_runner_instance(
            self.args, run_id=run_id, request_json=request_json,
            agent_config=self.agent_config, edge_id=edge_id
        )

        run_id_str = str(run_id)
        server_runner.edge_id = self.edge_id
        server_runner.server_agent_id = self.server_agent_id
        server_runner.start_request_json = json.dumps(request_json)
        self.run_process_event = multiprocessing.Event()
        server_runner.run_process_event = self.run_process_event
        self.run_process_completed_event = multiprocessing.Event()
        server_runner.run_process_completed_event = self.run_process_completed_event
        server_runner.edge_id_status_queue = self.run_edge_id_status_queue
        server_runner.edge_device_info_queue = self.run_edge_device_info_queue
        self.run_extend_queue_list = self._generate_extend_queue_list()
        if platform.system() == "Windows":
            self.run_process = multiprocessing.Process(
                target=server_runner.run if not is_server_job else server_runner.run_server_job, args=(
                    self.run_process_event, self.run_process_completed_event, self.run_edge_id_status_queue,
                    self.run_edge_device_info_queue, self.run_metrics_queue, self.run_events_queue,
                    self.run_artifacts_queue, self.run_logs_queue, self.run_edge_device_info_global_queue,
                    self.run_extend_queue_list, sender_message_queue, listener_message_queue, status_center_queue,
                    process_name,
                )
            )
        else:
            self.run_process = fedml.get_process(
                target=server_runner.run if not is_server_job else server_runner.run_server_job, args=(
                    self.run_process_event, self.run_process_completed_event, self.run_edge_id_status_queue,
                    self.run_edge_device_info_queue, self.run_metrics_queue, self.run_events_queue,
                    self.run_artifacts_queue, self.run_logs_queue, self.run_edge_device_info_global_queue,
                    self.run_extend_queue_list, sender_message_queue, listener_message_queue, status_center_queue,
                    process_name,
                )
            )
        self.run_process.start()
        ServerConstants.save_run_process(run_id, self.run_process.pid)
        return self.run_process

    def put_run_edge_device_info_to_queue(self, run_id, edge_id, device_info):
        edge_ids = self.request_json.get("edgeids", None)
        if edge_ids is None:
            return
        if int(edge_id) in edge_ids or str(edge_id) in edge_ids:
            run_id_str = str(run_id)
            if self.run_edge_device_info_queue is None:
                self.run_edge_device_info_queue = multiprocessing.Manager().Queue()
            self.run_edge_device_info_queue.put(device_info)

    def should_continue_run_job(self, run_id):
        run_config = self.request_json["run_config"]
        run_params = run_config.get("parameters", {})
        job_yaml = run_params.get("job_yaml", {})
        job_yaml_default_none = run_params.get("job_yaml", None)
        framework_type = job_yaml.get("framework_type", None)
        job_type = job_yaml.get("job_type", None)
        job_type = job_yaml.get("task_type", Constants.JOB_TASK_TYPE_TRAIN) if job_type is None else job_type
        if job_yaml_default_none is not None:
            if job_type == Constants.JOB_TASK_TYPE_FEDERATE:
                return True

            if framework_type is None or framework_type != Constants.JOB_FRAMEWORK_TYPE_FEDML:
                self.status_reporter.report_server_id_status(
                    run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING, edge_id=self.edge_id,
                    server_id=self.edge_id, server_agent_id=self.edge_id)
                return False

        return True

    @debug
    def detect_edges_status(
            self, edge_device_info_queue, edge_device_info_global_queue=None, callback_when_edges_ready=None,
            status_timeout=None,
            need_to_trigger_exception=True, status_check_context=None, given_edge_ids=None,
            callback_when_detecting=None, args_for_callback_when_detecting=None
    ):
        run_id = self.request_json["runId"]
        run_id_str = str(run_id)
        edge_id_list = self.request_json["edgeids"]
        if given_edge_ids is not None:
            edge_id_list = given_edge_ids

        # Init realtime status of all edges
        run_edges_realtime_status = dict()
        run_edges_realtime_status[run_id_str] = dict()

        total_sleep_seconds = 0
        status_check_sleep_seconds = 10
        allowed_status_check_sleep_seconds = 60 * 2 if status_timeout is None else status_timeout
        allowed_status_check_sleep_seconds_for_async = 30
        inactivate_edges = list()
        active_edge_info_dict = dict()
        while True:
            if callback_when_detecting is not None:
                callback_when_detecting(args_for_callback_when_detecting)

            # Fetch edge info from the edge status queue, which will be added to realtime status map
            while True:
                self.check_runner_stop_event()

                try:
                    edge_info = edge_device_info_queue.get(block=False, timeout=1)
                    if edge_info is not None:
                        edge_id = edge_info.get("edge_id", None)
                        if edge_id is not None:
                            run_edges_realtime_status[run_id_str][edge_id] = edge_info
                except queue.Empty as e:  # If queue is empty, then break loop
                    break

            self.check_runner_stop_event()

            # Check all edges which don't send response status successfully
            # and retry to send the status checking message.
            active_edges_count = 0
            inactivate_edges.clear()
            active_edge_info_dict.clear()
            for edge_id in edge_id_list:
                edge_info_dict = run_edges_realtime_status.get(run_id_str, {})
                edge_info = edge_info_dict.get(edge_id, None)
                edge_info = edge_info_dict.get(str(edge_id), None) if edge_info is None else edge_info
                if edge_info is not None:
                    active_edges_count += 1
                    active_edge_info_dict[str(edge_id)] = edge_info
                else:
                    inactivate_edges.append(edge_id)

            # If all edges are ready then send the starting job message to them
            if active_edges_count == len(edge_id_list):
                logging.info(f"All edges are ready. Active edge id list is as follows. {active_edge_info_dict}")
                if callback_when_edges_ready is not None:
                    logging.info("All edges are ready. Start to process the callback function.")
                    callback_when_edges_ready(self.request_json, active_edge_info_dict=active_edge_info_dict)
                else:
                    logging.info("All edges are ready. No callback function to process.")
                break
            else:
                logging.info(f"All edges are not ready. Active edge id list: {active_edge_info_dict}, "
                             f"Inactive edge id list: {inactivate_edges}")

            # Check if runner needs to stop and sleep specific time
            self.check_runner_stop_event()
            time.sleep(status_check_sleep_seconds)
            total_sleep_seconds += status_check_sleep_seconds

            # Check if the status response message has timed out to receive
            if total_sleep_seconds >= allowed_status_check_sleep_seconds:
                # If so, send failed message to MLOps and send exception message to all edges.
                logging.error(f"There are inactive edge devices. "
                              f"Inactivate edge id list is as follows. {inactivate_edges}")
                if need_to_trigger_exception:
                    self.status_reporter.report_server_id_status(
                        run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED, edge_id=self.edge_id,
                        server_id=self.edge_id, server_agent_id=self.server_agent_id)
                    self.report_exception_status(run_id)
                return False, active_edge_info_dict, inactivate_edges

            # If we enable the mode for async cluster, then sleep some time and send messages to all clients.
            if callback_when_edges_ready is not None and self.should_process_async_cluster is not None:
                should_async, async_timeout = self.should_process_async_cluster()
                if should_async and total_sleep_seconds >= allowed_status_check_sleep_seconds_for_async:
                    if async_timeout > allowed_status_check_sleep_seconds_for_async:
                        time.sleep(async_timeout - allowed_status_check_sleep_seconds_for_async)
                    self.send_training_request_to_edges(self.request_json, active_edge_info_dict)
                    return True, active_edge_info_dict, inactivate_edges

        return True, active_edge_info_dict, inactivate_edges

    def report_exception_status(self, run_id):
        self.mlops_metrics.report_job_status(run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_EXCEPTION)

    def callback_run_logs(self, topic, payload):
        run_id = str(topic).split('/')[-1]
        run_id_str = str(run_id)
        if self.run_logs_queue is None:
            self.run_logs_queue = multiprocessing.Manager().Queue()
        self.run_logs_queue.put(payload)

    def callback_run_metrics(self, topic, payload):
        print(f"callback_run_metrics topic {topic}, payload {payload}")
        run_id = str(topic).split('/')[-1]
        run_id_str = str(run_id)
        if self.run_metrics_queue is None:
            self.run_metrics_queue = multiprocessing.Manager().Queue()
        self.run_metrics_queue.put(payload)

    # def send_training_request_to_edges(self, active_edge_info_dict):
    #     topic = GeneralConstants.MSG_TOPIC_SEND_TRAINING_REQUEST_TO_EDGES
    #     payload = json.dumps(active_edge_info_dict)
    #     self.message_center.receive_message(topic, payload)
    def send_training_request_to_edges(self, request_json, active_edge_info_dict=None):
        run_id = request_json["runId"]
        edge_id_list = request_json["edgeids"]
        run_config = request_json.get("run_config", {})
        run_params = run_config.get("parameters", {})
        job_yaml = run_params.get("job_yaml", {})
        job_yaml_default_none = run_params.get("job_yaml", None)
        computing = job_yaml.get("computing", {})
        request_num_gpus = computing.get("minimum_num_gpus", None)
        job_gpu_id_list = request_json.get("job_gpu_id_list", None)
        assigned_gpu_num_dict = dict()
        assigned_gpu_ids_dict = dict()
        master_node_addr = ""
        master_node_port = 0

        logging.info(f"Send training request to Edge ids: {edge_id_list}, run_id {run_id}")

        should_match_gpu = False
        if job_yaml_default_none is not None and request_num_gpus is not None and \
                int(request_num_gpus) > 0 and active_edge_info_dict is not None:
            should_match_gpu = True
            SchedulerMatcher.parse_and_print_gpu_info_for_all_edges(active_edge_info_dict, show_gpu_list=True)

            # Match and assign gpus to each device
            assigned_gpu_num_dict, assigned_gpu_ids_dict = SchedulerMatcher.match_and_assign_gpu_resources_to_devices(
                request_num_gpus, edge_id_list, active_edge_info_dict, job_gpu_id_list=job_gpu_id_list)
            if assigned_gpu_num_dict is None or assigned_gpu_ids_dict is None:
                # If no resources available, send failed message to MLOps and send exception message to all edges.
                gpu_count, gpu_available_count = SchedulerMatcher.parse_and_print_gpu_info_for_all_edges(
                    active_edge_info_dict, should_print=True)
                err_info = f"No resources available." \
                           f"Total available GPU count {gpu_available_count} is less than " \
                           f"request GPU count {request_num_gpus}"
                logging.error(err_info)

                self.status_reporter.report_server_id_status(
                    run_id, GeneralConstants.MSG_MLOPS_SERVER_STATUS_FAILED, edge_id=self.edge_id,
                    server_id=self.edge_id, server_agent_id=self.server_agent_id)
                self.report_exception_status(run_id)

                serving_args = job_yaml.get("serving_args", {})
                endpoint_id = serving_args.get("endpoint_id", None)
                if endpoint_id is not None:
                    fedml.mlops.log_endpoint_status(
                        endpoint_id, GeneralConstants.MSG_MLOPS_SERVER_STATUS_FAILED)
                    fedml.mlops.log_run_log_lines(
                        endpoint_id, 0, [err_info],
                        log_source=GeneralConstants.FEDML_LOG_SOURCE_TYPE_MODEL_END_POINT
                    )
                return

            # Generate master node addr and port
            master_node_addr, master_node_port = SchedulerMatcher.get_master_node_info(edge_id_list,
                                                                                       active_edge_info_dict)

            # Generate new edge id list after matched
            edge_id_list = SchedulerMatcher.generate_new_edge_list_for_gpu_matching(assigned_gpu_num_dict)
            if len(edge_id_list) <= 0:
                gpu_count, gpu_available_count = SchedulerMatcher.parse_and_print_gpu_info_for_all_edges(
                    active_edge_info_dict, should_print=True)
                logging.error(f"Request parameter for GPU num is invalid."
                              f"Total available GPU count {gpu_available_count}."
                              f"Request GPU num {request_num_gpus}")
                self.status_reporter.report_server_id_status(
                    run_id, GeneralConstants.MSG_MLOPS_SERVER_STATUS_FAILED, edge_id=self.edge_id,
                    server_id=self.edge_id, server_agent_id=self.server_agent_id)
                self.report_exception_status(run_id)
                return

        if should_match_gpu:
            # Report gpu num and related infos to MLOps.
            serving_args = job_yaml.get("serving_args", {})
            endpoint_id = serving_args.get("endpoint_id", None)
            if endpoint_id is not None:
                endpoint_info = list()
                for edge_id_item, gpu_num in assigned_gpu_num_dict.items():
                    edge_info = active_edge_info_dict.get(str(edge_id_item), {})
                    endpoint_info.append({
                        "machine_id": edge_id_item, "endpoint_gpu_count": gpu_num,
                        "master_deploy_id": edge_info.get("master_device_id", 0),
                        "slave_deploy_id": edge_info.get("slave_device_id", 0)})
                topic_name = f"compute/mlops/endpoint"
                endpoint_info_json = {"endpoint_id": endpoint_id, "endpoint_info": endpoint_info}
                print(f"endpoint_info_json {endpoint_info_json}")
                self.message_center.send_message(topic_name, json.dumps(endpoint_info_json))

        client_rank = 1
        for edge_id in edge_id_list:
            topic_start_train = "flserver_agent/" + str(edge_id) + "/start_train"
            logging.info("start_train: send topic " + topic_start_train + " to client...")
            request_json["client_rank"] = client_rank
            client_rank += 1

            if active_edge_info_dict is not None:
                edge_info = active_edge_info_dict.get(str(edge_id), {})
                model_master_device_id = edge_info.get("master_device_id", None)
                model_slave_device_id = edge_info.get("slave_device_id", None)
                model_slave_device_id_list = edge_info.get("slave_device_id_list", None)

                if should_match_gpu:
                    request_json["scheduler_match_info"] = SchedulerMatcher.generate_match_info_for_scheduler(
                        edge_id, edge_id_list, master_node_addr, master_node_port,
                        assigned_gpu_num_dict, assigned_gpu_ids_dict,
                        model_master_device_id=model_master_device_id,
                        model_slave_device_id=model_slave_device_id,
                        model_slave_device_id_list=model_slave_device_id_list
                    )

            self.message_center.send_message(topic_start_train, json.dumps(request_json))

    def should_process_async_cluster(self):
        run_config = self.request_json.get("run_config", {})
        run_params = run_config.get("parameters", {})
        common_args = run_params.get("common_args", {})
        self.enable_async_cluster = common_args.get("enable_async_cluster", False)
        self.async_check_timeout = common_args.get("async_check_timeout", 0)
        if self.enable_async_cluster:
            return True, self.async_check_timeout

        return False, self.async_check_timeout

    def get_client_id_list(self, server_edge_id_list):
        return server_edge_id_list
