import json
import logging
import multiprocessing
import os
import platform
import time
import traceback
from abc import ABC, abstractmethod

import setproctitle

import fedml
from ....core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from ....core.mlops.mlops_runtime_log_daemon import MLOpsRuntimeLogDaemon
from .client_data_interface import FedMLClientDataInterface
from ..comm_utils import sys_utils
from ....core.mlops.mlops_utils import MLOpsUtils
from multiprocessing import Process
from ..scheduler_core.scheduler_base_job_runner import FedMLSchedulerBaseJobRunner, RunnerError, RunnerCompletedError
from ..scheduler_core.general_constants import GeneralConstants
from ..comm_utils.job_utils import JobRunnerUtils


class FedMLBaseSlaveJobRunner(FedMLSchedulerBaseJobRunner, ABC):

    def __init__(self, args, edge_id=0, request_json=None, agent_config=None, run_id=0,
                 cuda_visible_gpu_ids_str=None,
                 agent_data_dir=None, agent_package_download_dir=None,
                 agent_package_unzip_dir=None, agent_log_file_dir=None):
        FedMLSchedulerBaseJobRunner.__init__(
            self, args, edge_id=edge_id, request_json=request_json, agent_config=agent_config, run_id=run_id,
            cuda_visible_gpu_ids_str=cuda_visible_gpu_ids_str, agent_data_dir=agent_data_dir,
            agent_package_download_dir=agent_package_download_dir,
            agent_package_unzip_dir=agent_package_unzip_dir,
            agent_log_file_dir=agent_log_file_dir
        )

        self.fedml_data_base_package_dir = os.path.join("/", "fedml", "data")
        self.fedml_data_local_package_dir = os.path.join("/", "fedml", "fedml-package", "fedml", "data")
        self.fedml_data_dir = self.fedml_data_base_package_dir
        self.fedml_config_dir = os.path.join("/", "fedml", "conf")
        self.run_extend_queue_list = None
        self.computing_started_time = 0

    def __repr__(self):
        return "<{klass} @{id:x} {attrs}>".format(
            klass=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=" ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def run(self, process_event, completed_event,  run_extend_queue_list,
            sender_message_center, listener_message_queue, status_center_queue,
            process_name=None):
        if process_name is not None:
            setproctitle.setproctitle(process_name)

        print(f"Client runner process id {os.getpid()}, name {process_name}, run id {self.run_id}")

        if platform.system() != "Windows":
            os.setsid()

        os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
        os.environ.setdefault('PYTHONWARNINGS', 'ignore:semaphore_tracker:UserWarning')

        self.run_process_event = process_event
        self.run_process_completed_event = completed_event
        try:
            MLOpsUtils.set_ntp_offset(self.ntp_offset)
            self.rebuild_message_status_center(sender_message_center, listener_message_queue, status_center_queue)
            self.run_impl(run_extend_queue_list, sender_message_center, listener_message_queue, status_center_queue)
        except RunnerError:
            logging.info("Runner stopped.")
            self.reset_devices_status(self.edge_id, GeneralConstants.MSG_MLOPS_CLIENT_STATUS_KILLED)
        except RunnerCompletedError:
            logging.info("Runner completed.")
        except Exception as e:
            logging.error(f"Runner exited with errors. Exception: {e}, Traceback {traceback.format_exc()}")
            self.status_reporter.report_client_id_status(
                self.edge_id, GeneralConstants.MSG_MLOPS_CLIENT_STATUS_FAILED,
                is_from_model=self.is_deployment_runner, server_id=self.server_id, run_id=self.run_id)
        finally:
            if self.mlops_metrics is not None:
                computing_ended_time = MLOpsUtils.get_ntp_time()
                self.mlops_metrics.report_edge_job_computing_cost(self.run_id, self.edge_id,
                                                                  self.computing_started_time, computing_ended_time,
                                                                  self.args.account_id, self.args.api_key)
            logging.info("Release resources.")
            job_type = JobRunnerUtils.parse_job_type(self.request_json)
            FedMLSchedulerBaseJobRunner.cleanup_containers_and_release_gpus(self.run_id, self.edge_id, job_type)
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(self.run_id, self.edge_id)
            if self.mlops_metrics is not None:
                self.mlops_metrics.stop_sys_perf()
            time.sleep(3)
            GeneralConstants.cleanup_learning_process(self.run_id)
            GeneralConstants.cleanup_run_process(self.run_id)

    @abstractmethod
    def run_impl(self, run_extend_queue_list, sender_message_center,
                 listener_message_queue, status_center_queue):
        run_id = self.request_json["runId"]
        run_config = self.request_json["run_config"]
        data_config = run_config.get("data_config", {})
        packages_config = run_config["packages_config"]

        self.computing_started_time = MLOpsUtils.get_ntp_time()
        self.mlops_metrics.report_edge_job_computing_cost(run_id, self.edge_id,
                                                          self.computing_started_time, 0,
                                                          self.args.account_id, self.args.api_key)

        self.check_runner_stop_event()

        MLOpsRuntimeLog.get_instance(self.args).init_logs(log_level=logging.INFO)

        self.status_reporter.report_client_id_status(
            self.edge_id, GeneralConstants.MSG_MLOPS_CLIENT_STATUS_INITIALIZING,
            is_from_model=self.is_deployment_runner, running_json=json.dumps(self.request_json), run_id=run_id)

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

        logging.info("Download packages")

        # update local config with real time parameters from server and dynamically replace variables value
        unzip_package_path, fedml_config_object = self.update_local_fedml_config(run_id, run_config)
        # if unzip_package_path is None or fedml_config_object is None:
        #     logging.info("failed to update local fedml config.")
        #     self.check_runner_stop_event()
        #     # Send failed msg when exceptions.
        #     self.cleanup_run_when_starting_failed(status=GeneralConstants.MSG_MLOPS_CLIENT_STATUS_EXCEPTION)
        #     return

        logging.info("Check downloaded packages...")

        entry_file_config = fedml_config_object.get("entry_config", None)
        dynamic_args_config = fedml_config_object.get("dynamic_args", None)
        entry_file = str(entry_file_config["entry_file"]).replace('\\', os.sep).replace('/', os.sep)
        entry_file = os.path.basename(entry_file)
        conf_file = entry_file_config["conf_file"]
        conf_file = str(conf_file).replace('\\', os.sep).replace('/', os.sep)
        #####
        # GeneralConstants.cleanup_learning_process(run_id)
        # GeneralConstants.cleanup_bootstrap_process(run_id)
        #####

        if not os.path.exists(unzip_package_path):
            logging.info("failed to unzip file.")
            self.check_runner_stop_event()
            return
        os.chdir(os.path.join(unzip_package_path, "fedml"))

        self.check_runner_stop_event()

        logging.info("starting the user process...")

        entry_file_full_path = os.path.join(unzip_package_path, "fedml", entry_file)
        conf_file_full_path = os.path.join(unzip_package_path, "fedml", conf_file)
        logging.info("waiting the user process to finish...")
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

        ret_code, out, err = process.returncode if process else None, None, None
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

                self.status_reporter.report_client_id_status(
                    self.edge_id, GeneralConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED,
                    is_from_model=self.is_deployment_runner, server_id=self.server_id, run_id=run_id)

                if is_launch_task:
                    sys_utils.log_return_info(f"job {run_id}", ret_code)
                else:
                    sys_utils.log_return_info(entry_file, ret_code)
        else:
            is_run_ok = False

        if not is_run_ok:
            # If the run status is killed or finished, then return with the normal state.
            current_job = FedMLClientDataInterface.get_instance().get_job_by_id(run_id)
            if current_job is not None and (current_job.status == GeneralConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED or
                                            current_job.status == GeneralConstants.MSG_MLOPS_CLIENT_STATUS_KILLED):
                return

            self.check_runner_stop_event()

            logging.error("failed to run the learning process...")

            if err is not None:
                err_str = sys_utils.decode_our_err_result(err)
                if err_str != "":
                    logging.error("{}".format(err_str))

            if is_launch_task:
                sys_utils.log_return_info(f"job {run_id}", ret_code)
            else:
                sys_utils.log_return_info(entry_file, ret_code)

            # Send failed msg when exceptions.
            self.status_reporter.report_client_id_status(
                self.edge_id, GeneralConstants.MSG_MLOPS_CLIENT_STATUS_FAILED,
                is_from_model=self.is_deployment_runner, server_id=self.server_id, run_id=run_id)

    @abstractmethod
    def _generate_job_runner_instance(self, args, run_id=None, request_json=None, agent_config=None, edge_id=None):
        return None

    @abstractmethod
    def _generate_extend_queue_list(self):
        return list()

    def reset_devices_status(self, edge_id, status):
        self.status_reporter.run_id = self.run_id
        self.status_reporter.edge_id = edge_id
        self.status_reporter.report_client_id_status(
            edge_id, status, is_from_model=self.is_deployment_runner, server_id=self.server_id, run_id=self.run_id)

    def start_runner_process(
            self, run_id, request_json, edge_id=None,
            sender_message_queue=None, listener_message_queue=None,
            status_center_queue=None, cuda_visible_gpu_ids_str=None, process_name=None
    ):
        client_runner = self._generate_job_runner_instance(
            self.args, run_id=run_id, request_json=request_json,
            agent_config=None, edge_id=edge_id
        )
        client_runner.start_request_json = request_json
        run_id_str = str(run_id)
        self.run_process_event = multiprocessing.Event()
        client_runner.run_process_event = self.run_process_event
        self.run_process_completed_event = multiprocessing.Event()
        client_runner.run_process_completed_event = self.run_process_completed_event
        client_runner.server_id = request_json.get("server_id", "0")
        self.run_extend_queue_list = self._generate_extend_queue_list()
        logging.info("start the runner process.")

        if platform.system() == "Windows":
            self.run_process = multiprocessing.Process(
                target=client_runner.run, args=(
                    self.run_process_event, self.run_process_completed_event, self.run_extend_queue_list,
                    sender_message_queue, listener_message_queue, status_center_queue, process_name
                ))
        else:
            self.run_process = fedml.get_process(target=client_runner.run, args=(
                self.run_process_event, self.run_process_completed_event, self.run_extend_queue_list,
                sender_message_queue, listener_message_queue, status_center_queue, process_name
            ))
        self.run_process.start()
        return self.run_process
