import json
import logging
import multiprocessing
import sys

from multiprocessing import Process
import os
import platform
import shutil
import subprocess
import threading

import time
import traceback
import urllib
import uuid
import zipfile
from urllib.parse import urljoin, urlparse

import requests

import fedml
from ..comm_utils.constants import SchedulerConstants
from ..comm_utils.job_cleanup import JobCleanup
from ..comm_utils.job_utils import JobRunnerUtils, DockerArgs
from ..comm_utils.run_process_utils import RunProcessUtils
from ..scheduler_entry.constants import Constants
from ....core.mlops.mlops_device_perfs import MLOpsDevicePerfStats
from ....core.mlops.mlops_runtime_log import MLOpsRuntimeLog

from ....core.distributed.communication.mqtt.mqtt_manager import MqttManager
from ..comm_utils.yaml_utils import load_yaml_config
from .client_constants import ClientConstants

from ....core.mlops.mlops_metrics import MLOpsMetrics

from ....core.mlops.mlops_configs import MLOpsConfigs
from ....core.mlops.mlops_runtime_log_daemon import MLOpsRuntimeLogDaemon
from ....core.mlops.mlops_status import MLOpsStatus
from ..comm_utils.sys_utils import get_sys_runner_info, get_python_program
from .client_data_interface import FedMLClientDataInterface
from ..comm_utils import sys_utils
from ....core.mlops.mlops_utils import MLOpsUtils
from ..model_scheduler.model_device_client import FedMLModelDeviceClientRunner
from ..model_scheduler.model_device_server import FedMLModelDeviceServerRunner
from ..comm_utils import security_utils
from ..scheduler_core.compute_cache_manager import ComputeCacheManager
from ..scheduler_core.message_center import FedMLMessageCenter


class RunnerError(Exception):
    """ Runner stopped. """
    pass


class RunnerCompletedError(Exception):
    """ Runner completed. """
    pass


class FedMLClientRunner(FedMLMessageCenter):

    def __init__(self, args, edge_id=0, request_json=None, agent_config=None, run_id=0,
                 cuda_visible_gpu_ids_str=None):
        super().__init__()
        self.model_device_server_id = None
        self.model_device_client_edge_id_list = None
        self.disable_client_login = False
        self.model_device_server = None
        self.model_device_client_list = None
        self.run_process_event = None
        self.run_process_event_map = dict()
        self.run_process_completed_event = None
        self.run_process_completed_event_map = dict()
        self.run_process = None
        self.run_process_map = dict()
        self.running_request_json = dict()
        self.local_api_process = None
        self.start_request_json = None
        self.device_status = None
        self.current_training_status = None
        self.mqtt_mgr = None
        self.edge_id = edge_id
        self.edge_user_name = None
        self.edge_extra_url = None
        self.run_id = run_id
        self.unique_device_id = None
        self.args = args
        self.request_json = request_json
        self.version = args.version
        self.device_id = args.device_id
        self.cur_dir = os.path.split(os.path.realpath(__file__))[0]
        if args.current_running_dir is not None:
            self.cur_dir = args.current_running_dir
        self.sudo_cmd = ""
        self.is_mac = False
        if platform.system() == "Darwin":
            self.is_mac = True

        self.agent_config = agent_config
        self.fedml_data_base_package_dir = os.path.join("/", "fedml", "data")
        self.fedml_data_local_package_dir = os.path.join("/", "fedml", "fedml-package", "fedml", "data")
        self.fedml_data_dir = self.fedml_data_base_package_dir
        self.fedml_config_dir = os.path.join("/", "fedml", "conf")

        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES = {
            "${FEDSYS.RUN_ID}": "",
            "${FEDSYS.PRIVATE_LOCAL_DATA}": "",
            "${FEDSYS.CLIENT_ID_LIST}": "",
            "${FEDSYS.SYNTHETIC_DATA_URL}": "",
            "${FEDSYS.IS_USING_LOCAL_DATA}": "",
            "${FEDSYS.CLIENT_NUM}": "",
            "${FEDSYS.CLIENT_INDEX}": "",
            "${FEDSYS.CLIENT_OBJECT_LIST}": "",
            "${FEDSYS.LOG_SERVER_URL}": "",
        }

        self.mlops_metrics = None
        self.client_active_list = dict()
        self.ntp_offset = MLOpsUtils.get_ntp_offset()
        self.server_id = None
        self.computing_started_time = 0
        self.fedml_config_object = None
        self.package_type = SchedulerConstants.JOB_PACKAGE_TYPE_DEFAULT
        self.cuda_visible_gpu_ids_str = cuda_visible_gpu_ids_str
        # logging.info("Current directory of client agent: " + self.cur_dir)
        self.subscribed_topics = list()
        self.user_name = None
        self.general_edge_id = None
        self.message_center = None

    def __repr__(self):
        return "<{klass} @{id:x} {attrs}>".format(
            klass=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=" ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def copy_runner(self):
        copy_runner = FedMLClientRunner(self.args)
        copy_runner.disable_client_login =  self.disable_client_login
        copy_runner.model_device_server = self.model_device_server
        copy_runner.model_device_client_list = self.model_device_client_list
        copy_runner.run_process_event = self.run_process_event
        copy_runner.run_process_event_map = self.run_process_event_map
        copy_runner.run_process_completed_event = self.run_process_completed_event
        copy_runner.run_process_completed_event_map = self.run_process_completed_event_map
        copy_runner.run_process = self.run_process
        copy_runner.run_process_map = self.run_process_map
        copy_runner.running_request_json = self.running_request_json
        copy_runner.local_api_process = self.local_api_process
        copy_runner.start_request_json = self.start_request_json
        copy_runner.device_status = self.device_status
        copy_runner.current_training_status = self.current_training_status
        copy_runner.mqtt_mgr = self.mqtt_mgr
        copy_runner.edge_id = self.edge_id
        copy_runner.edge_user_name = self.edge_user_name
        copy_runner.edge_extra_url = self.edge_extra_url
        copy_runner.run_id = self.run_id
        copy_runner.unique_device_id = self.unique_device_id
        copy_runner.args = self.args
        copy_runner.request_json = self.request_json
        copy_runner.version =self.version
        copy_runner.device_id = self.device_id
        copy_runner.cur_dir = self.cur_dir
        copy_runner.cur_dir = self.cur_dir
        copy_runner.sudo_cmd = self.sudo_cmd
        copy_runner.is_mac = self.is_mac

        copy_runner.agent_config = self.agent_config
        copy_runner.fedml_data_base_package_dir = self.fedml_data_base_package_dir
        copy_runner.fedml_data_local_package_dir = self.fedml_data_local_package_dir
        copy_runner.fedml_data_dir = self.fedml_data_dir
        copy_runner.fedml_config_dir = self.fedml_config_dir

        copy_runner.FEDML_DYNAMIC_CONSTRAIN_VARIABLES = self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES

        copy_runner.mlops_metrics = self.mlops_metrics
        copy_runner.client_active_list = self.client_active_list
        copy_runner.ntp_offset = self.ntp_offset
        copy_runner.server_id = self.server_id
        copy_runner.computing_started_time = self.computing_started_time
        copy_runner.fedml_config_object = self.fedml_config_object
        copy_runner.package_type = self.package_type
        copy_runner.cuda_visible_gpu_ids_str = self.cuda_visible_gpu_ids_str
        copy_runner.subscribed_topics = self.subscribed_topics
        copy_runner.user_name = self.user_name
        copy_runner.general_edge_id = self.general_edge_id
        copy_runner.message_center = self.message_center

        return copy_runner

    def build_dynamic_constrain_variables(self, run_id, run_config):
        data_config = run_config.get("data_config", {})
        server_edge_id_list = self.request_json["edgeids"]
        local_edge_id_list = list()
        local_edge_id_list.append(int(self.edge_id))
        is_using_local_data = 0
        private_data_dir = data_config.get("privateLocalData", "")
        synthetic_data_url = data_config.get("syntheticDataUrl", "")
        edges = self.request_json["edges"]
        # if private_data_dir is not None \
        #         and len(str(private_data_dir).strip(' ')) > 0:
        #     is_using_local_data = 1
        if private_data_dir is None or len(str(private_data_dir).strip(" ")) <= 0:
            params_config = run_config.get("parameters", None)
            private_data_dir = ClientConstants.get_data_dir()
        if synthetic_data_url is None or len(str(synthetic_data_url)) <= 0:
            synthetic_data_url = private_data_dir

        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.RUN_ID}"] = run_id
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.PRIVATE_LOCAL_DATA}"] = private_data_dir.replace(" ", "")
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.CLIENT_ID_LIST}"] = str(local_edge_id_list).replace(" ", "")
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.SYNTHETIC_DATA_URL}"] = synthetic_data_url.replace(" ", "")
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.IS_USING_LOCAL_DATA}"] = str(is_using_local_data)
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.CLIENT_NUM}"] = len(server_edge_id_list)
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.CLIENT_INDEX}"] = 1
        for cur_index, id_value in enumerate(server_edge_id_list):
            if str(id_value) == str(self.edge_id):
                self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.CLIENT_INDEX}"] = cur_index + 1
                break
        client_objects = str(json.dumps(edges))
        client_objects = client_objects.replace(" ", "").replace("\n", "").replace('"', '\\"')
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.CLIENT_OBJECT_LIST}"] = client_objects
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.LOG_SERVER_URL}"] = self.agent_config["ml_ops_config"][
            "LOG_SERVER_URL"
        ]

    def unzip_file(self, zip_file, unzip_file_path) -> str:
        if zipfile.is_zipfile(zip_file):
            with zipfile.ZipFile(zip_file, "r") as zipf:
                zipf.extractall(unzip_file_path)
                unzipped_file_name = zipf.namelist()[0]
        else:
            raise Exception("Invalid zip file {}".format(zip_file))

        return unzipped_file_name

    def package_download_progress(self, count, blksize, filesize):
        self.check_runner_stop_event()

        downloaded = count * blksize
        downloaded = filesize if downloaded > filesize else downloaded
        progress = (downloaded / filesize * 100) if filesize != 0 else 0
        progress_int = int(progress)
        downloaded_kb = format(downloaded / 1024, '.2f')

        # since this hook funtion is stateless, we need a state to avoid print progress repeatly
        if count == 0:
            self.prev_download_progress = 0
        if progress_int != self.prev_download_progress and progress_int % 5 == 0:
            self.prev_download_progress = progress_int
            logging.info("package downloaded size {} KB, progress {}%".format(downloaded_kb, progress_int))

    def retrieve_and_unzip_package(self, package_name, package_url):
        local_package_path = ClientConstants.get_package_download_dir()
        os.makedirs(local_package_path, exist_ok=True)
        filename, filename_without_extension, file_extension = ClientConstants.get_filename_and_extension(package_url)
        local_package_file = os.path.join(local_package_path, f"fedml_run_{self.run_id}_{filename_without_extension}")
        if os.path.exists(local_package_file):
            os.remove(local_package_file)
        package_url_without_query_path = urljoin(package_url, urlparse(package_url).path)
        urllib.request.urlretrieve(package_url_without_query_path, local_package_file,
                                   reporthook=self.package_download_progress)
        unzip_package_path = os.path.join(ClientConstants.get_package_unzip_dir(),
                                          f"unzip_fedml_run_{self.run_id}_{filename_without_extension}")
        try:
            shutil.rmtree(unzip_package_path, ignore_errors=True)
        except Exception as e:
            logging.error(
                f"Failed to remove directory {unzip_package_path}, Exception: {e}, Traceback: {traceback.format_exc()}")
            pass

        package_dir_name = self.unzip_file(local_package_file, unzip_package_path)  # Using unziped folder name
        unzip_package_full_path = os.path.join(unzip_package_path, package_dir_name)

        logging.info("local_package_file {}, unzip_package_path {}, unzip file full path {}".format(
            local_package_file, unzip_package_path, unzip_package_full_path))

        return unzip_package_full_path

    def update_local_fedml_config(self, run_id, run_config):
        packages_config = run_config["packages_config"]

        # Copy config file from the client
        unzip_package_path = self.retrieve_and_unzip_package(
            packages_config["linuxClient"], packages_config["linuxClientUrl"]
        )
        fedml_local_config_file = os.path.join(unzip_package_path, "conf", "fedml.yaml")

        # Load the above config to memory
        config_from_container = load_yaml_config(fedml_local_config_file)
        container_entry_file_config = config_from_container["entry_config"]
        container_dynamic_args_config = config_from_container["dynamic_args"]
        entry_file = container_entry_file_config["entry_file"]
        conf_file = container_entry_file_config["conf_file"]
        self.package_type = container_entry_file_config.get("package_type", SchedulerConstants.JOB_PACKAGE_TYPE_DEFAULT)
        full_conf_path = os.path.join(unzip_package_path, "fedml", "config", os.path.basename(conf_file))

        # Dynamically build constrain variable with realtime parameters from server
        self.build_dynamic_constrain_variables(run_id, run_config)

        # Update entry arguments value with constrain variable values with realtime parameters from server
        # currently we support the following constrain variables:
        # ${FEDSYS_RUN_ID}: a run id represented one entire Federated Learning flow
        # ${FEDSYS_PRIVATE_LOCAL_DATA}: private local data path in the Federated Learning client
        # ${FEDSYS_CLIENT_ID_LIST}: client list in one entire Federated Learning flow
        # ${FEDSYS_SYNTHETIC_DATA_URL}: synthetic data url from server,
        #                  if this value is not null, the client will download data from this URL to use it as
        #                  federated training data set
        # ${FEDSYS_IS_USING_LOCAL_DATA}: whether use private local data as federated training data set
        # container_dynamic_args_config["data_cache_dir"] = "${FEDSYS.PRIVATE_LOCAL_DATA}"
        for constrain_variable_key, constrain_variable_value in self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES.items():
            for argument_key, argument_value in container_dynamic_args_config.items():
                if argument_value is not None and str(argument_value).find(constrain_variable_key) == 0:
                    replaced_argument_value = str(argument_value).replace(
                        constrain_variable_key, str(constrain_variable_value)
                    )
                    container_dynamic_args_config[argument_key] = replaced_argument_value

        # Merge all container new config sections as new config dictionary
        package_conf_object = dict()
        package_conf_object["entry_config"] = container_entry_file_config
        package_conf_object["dynamic_args"] = container_dynamic_args_config
        package_conf_object["dynamic_args"]["config_version"] = self.args.config_version
        container_dynamic_args_config["mqtt_config_path"] = os.path.join(
            unzip_package_path, "fedml", "config", os.path.basename(container_dynamic_args_config["mqtt_config_path"])
        )
        container_dynamic_args_config["s3_config_path"] = os.path.join(
            unzip_package_path, "fedml", "config", os.path.basename(container_dynamic_args_config["s3_config_path"])
        )
        log_file_dir = ClientConstants.get_log_file_dir()
        os.makedirs(log_file_dir, exist_ok=True)
        package_conf_object["dynamic_args"]["log_file_dir"] = log_file_dir

        # Save new config dictionary to local file
        fedml_updated_config_file = os.path.join(unzip_package_path, "conf", "fedml.yaml")
        ClientConstants.generate_yaml_doc(package_conf_object, fedml_updated_config_file)

        # Build dynamic arguments and set arguments to fedml config object
        self.build_dynamic_args(run_id, run_config, package_conf_object, unzip_package_path)
        return unzip_package_path, package_conf_object

    def build_dynamic_args(self, run_id, run_config, package_conf_object, base_dir):
        fedml_conf_file = package_conf_object["entry_config"]["conf_file"]
        fedml_conf_file_processed = str(fedml_conf_file).replace('\\', os.sep).replace('/', os.sep)
        fedml_conf_path = os.path.join(base_dir, "fedml", "config",
                                       os.path.basename(fedml_conf_file_processed))
        fedml_conf_object = load_yaml_config(fedml_conf_path)
        run_params = run_config.get("parameters", {})
        job_yaml = run_params.get("job_yaml", {})

        # Replace local fedml config objects with parameters from MLOps web
        parameters_object = run_config.get("parameters", None)
        if parameters_object is not None:
            for config_k, config_v in fedml_conf_object.items():
                parameter_v = parameters_object.get(config_k, None)
                if parameter_v is not None:
                    fedml_conf_object[config_k] = parameter_v
                    parameters_object.pop(config_k)

            for config_k, config_v in parameters_object.items():
                fedml_conf_object[config_k] = config_v

        package_dynamic_args = package_conf_object["dynamic_args"]
        if fedml_conf_object.get("comm_args", None) is not None:
            fedml_conf_object["comm_args"]["mqtt_config_path"] = package_dynamic_args["mqtt_config_path"]
            fedml_conf_object["comm_args"]["s3_config_path"] = package_dynamic_args["s3_config_path"]
            fedml_conf_object["common_args"]["using_mlops"] = True
        if fedml_conf_object.get("train_args", None) is not None:
            fedml_conf_object["train_args"]["run_id"] = package_dynamic_args["run_id"]
            fedml_conf_object["train_args"]["client_id_list"] = package_dynamic_args["client_id_list"]
            fedml_conf_object["train_args"]["client_num_in_total"] = int(package_dynamic_args["client_num_in_total"])
            fedml_conf_object["train_args"]["client_num_per_round"] = int(package_dynamic_args["client_num_in_total"])
            fedml_conf_object["train_args"]["client_id"] = self.edge_id
            fedml_conf_object["train_args"]["server_id"] = self.request_json.get("server_id", "0")
        if fedml_conf_object.get("device_args", None) is not None:
            fedml_conf_object["device_args"]["worker_num"] = int(package_dynamic_args["client_num_in_total"])
        # fedml_conf_object["data_args"]["data_cache_dir"] = package_dynamic_args["data_cache_dir"]
        data_args = fedml_conf_object.get("data_args")
        if data_args is not None:
            data_cache_dir = fedml_conf_object["data_args"].get("data_cache_dir")
            if data_cache_dir is not None:
                data_cache_dir = os.path.join(data_cache_dir, str(self.edge_id))
                fedml_conf_object["data_args"]["data_cache_dir"] = data_cache_dir
        if fedml_conf_object.get("tracking_args", None) is not None:
            fedml_conf_object["tracking_args"]["log_file_dir"] = package_dynamic_args["log_file_dir"]
            fedml_conf_object["tracking_args"]["log_server_url"] = package_dynamic_args["log_server_url"]

        fedml_conf_object["dynamic_args"] = package_dynamic_args
        self.fedml_config_object = fedml_conf_object.copy()
        ClientConstants.generate_yaml_doc(fedml_conf_object, fedml_conf_path)

    def run_bootstrap_script(self, bootstrap_cmd_list, bootstrap_script_file):
        try:
            logging.info("Bootstrap commands are being executed...")
            process, error_list = ClientConstants.execute_commands_with_live_logs(bootstrap_cmd_list,
                                                                                  callback=self.callback_run_bootstrap)

            ret_code, out, err = process.returncode, None, None
            if ret_code is None or ret_code <= 0:
                if error_list is not None and len(error_list) > 0:
                    is_bootstrap_run_ok = False
                else:
                    if out is not None:
                        out_str = sys_utils.decode_our_err_result(out)
                        if out_str != "":
                            logging.info("{}".format(out_str))

                    sys_utils.log_return_info(bootstrap_script_file, 0)

                    is_bootstrap_run_ok = True
            else:
                if err is not None:
                    err_str = sys_utils.decode_our_err_result(err)
                    if err_str != "":
                        logging.error("{}".format(err_str))

                sys_utils.log_return_info(bootstrap_script_file, ret_code)

                is_bootstrap_run_ok = False
        except Exception as e:
            logging.error(f"Bootstrap script error: Exception: {e}, Traceback: {traceback.format_exc()}")
            is_bootstrap_run_ok = False
        return is_bootstrap_run_ok

    def callback_run_bootstrap(self, job_pid):
        ClientConstants.save_bootstrap_process(self.run_id, job_pid)

    def run(self, process_event, completed_event, message_center_queue):
        print(f"Client runner process id {os.getpid()}, run id {self.run_id}")

        if platform.system() != "Windows":
            os.setsid()

        os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
        os.environ.setdefault('PYTHONWARNINGS', 'ignore:semaphore_tracker:UserWarning')

        self.run_process_event = process_event
        self.run_process_completed_event = completed_event
        try:
            MLOpsUtils.set_ntp_offset(self.ntp_offset)
            self.rebuild_message_center(message_center_queue)
            self.run_impl()
        except RunnerError:
            logging.info("Runner stopped.")
            self.reset_devices_status(self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED)
        except RunnerCompletedError:
            logging.info("Runner completed.")
        except Exception as e:
            logging.error(f"Runner exited with errors. Exception: {e}, Traceback {traceback.format_exc()}")
            self.mlops_metrics.report_client_id_status(
                self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED,
                server_id=self.server_id, run_id=self.run_id)
        finally:
            if self.mlops_metrics is not None:
                computing_ended_time = MLOpsUtils.get_ntp_time()
                self.mlops_metrics.report_edge_job_computing_cost(self.run_id, self.edge_id,
                                                                  self.computing_started_time, computing_ended_time,
                                                                  self.args.user, self.args.api_key)
            logging.info("Release resources.")
            self.cleanup_containers_and_release_gpus(self.run_id, self.edge_id)
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(self.run_id, self.edge_id)
            if self.mlops_metrics is not None:
                self.mlops_metrics.stop_sys_perf()
            time.sleep(3)
            ClientConstants.cleanup_learning_process(self.run_id)
            ClientConstants.cleanup_run_process(self.run_id)

    def check_runner_stop_event(self):
        if self.run_process_event.is_set():
            logging.info("Received stopping event.")
            raise RunnerError("Runner stopped")

        if self.run_process_completed_event.is_set():
            logging.info("Received completed event.")
            raise RunnerCompletedError("Runner completed")

    def run_impl(self):
        run_id = self.request_json["runId"]
        run_config = self.request_json["run_config"]
        data_config = run_config.get("data_config", {})
        packages_config = run_config["packages_config"]

        self.computing_started_time = MLOpsUtils.get_ntp_time()
        self.mlops_metrics.report_edge_job_computing_cost(run_id, self.edge_id,
                                                          self.computing_started_time, 0,
                                                          self.args.user, self.args.api_key)

        self.check_runner_stop_event()

        MLOpsRuntimeLog.get_instance(self.args).init_logs(log_level=logging.INFO)

        self.mlops_metrics.report_client_id_status(
            self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_INITIALIZING,
            running_json=self.start_request_json, run_id=run_id)

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
        #     self.cleanup_run_when_starting_failed(status=ClientConstants.MSG_MLOPS_CLIENT_STATUS_EXCEPTION)
        #     return

        logging.info("Check downloaded packages...")

        entry_file_config = fedml_config_object["entry_config"]
        dynamic_args_config = fedml_config_object["dynamic_args"]
        entry_file = str(entry_file_config["entry_file"]).replace('\\', os.sep).replace('/', os.sep)
        entry_file = os.path.basename(entry_file)
        conf_file = entry_file_config["conf_file"]
        conf_file = str(conf_file).replace('\\', os.sep).replace('/', os.sep)
        #####
        # ClientConstants.cleanup_learning_process(run_id)
        # ClientConstants.cleanup_bootstrap_process(run_id)
        #####

        if not os.path.exists(unzip_package_path):
            logging.info("failed to unzip file.")
            self.check_runner_stop_event()
            # Send failed msg when exceptions.
            self.cleanup_run_when_starting_failed(status=ClientConstants.MSG_MLOPS_CLIENT_STATUS_EXCEPTION)
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

        process, is_launch_task, error_list = self.execute_job_task(unzip_package_path=unzip_package_path,
                                                                    entry_file_full_path=entry_file_full_path,
                                                                    conf_file_full_path=conf_file_full_path,
                                                                    dynamic_args_config=dynamic_args_config,
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

                self.mlops_metrics.report_client_id_status(
                    self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED,
                    server_id=self.server_id, run_id=run_id)

                if is_launch_task:
                    sys_utils.log_return_info(f"job {run_id}", ret_code)
                else:
                    sys_utils.log_return_info(entry_file, ret_code)
        else:
            is_run_ok = False

        if not is_run_ok:
            # If the run status is killed or finished, then return with the normal state.
            current_job = FedMLClientDataInterface.get_instance().get_job_by_id(run_id)
            if current_job is not None and (current_job.status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED or
                                            current_job.status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED):
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
            self.mlops_metrics.report_client_id_status(
                self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED,
                server_id=self.server_id, run_id=run_id)

    def execute_job_task(self, unzip_package_path, entry_file_full_path, conf_file_full_path, dynamic_args_config,
                         fedml_config_object):
        run_config = self.request_json["run_config"]
        run_params = run_config.get("parameters", {})
        client_rank = self.request_json.get("client_rank", 1)
        job_yaml = run_params.get("job_yaml", {})
        job_yaml_default_none = run_params.get("job_yaml", None)
        job_api_key = job_yaml.get("run_api_key", None)
        job_api_key = job_yaml.get("fedml_run_dynamic_params", None) if job_api_key is None else job_api_key
        assigned_gpu_ids = run_params.get("gpu_ids", None)
        job_type = job_yaml.get("job_type", None)
        containerize = fedml_config_object.get("containerize", None)
        image_pull_policy = fedml_config_object.get("image_pull_policy", Constants.IMAGE_PULL_POLICY_ALWAYS)
        # TODO: Can we remove task_type?
        job_type = job_yaml.get("task_type", Constants.JOB_TASK_TYPE_TRAIN) if job_type is None else job_type
        conf_file_object = load_yaml_config(conf_file_full_path)
        entry_args_dict = conf_file_object.get("fedml_entry_args", {})
        entry_args = entry_args_dict.get("arg_items", None)
        scheduler_match_info = self.request_json.get("scheduler_match_info", {})
        if job_type == Constants.JOB_TASK_TYPE_TRAIN:
            containerize = True if containerize is None else containerize

        # Bootstrap Info
        bootstrap_script_path, bootstrap_script_dir, bootstrap_script_file = [None] * 3
        env_args = fedml_config_object.get("environment_args", None)

        if env_args is not None:
            bootstrap_script_file = env_args.get("bootstrap", None)
            if bootstrap_script_file is not None:
                bootstrap_script_file = str(bootstrap_script_file).replace('\\', os.sep).replace('/', os.sep)
                if platform.system() == 'Windows':
                    bootstrap_script_file = bootstrap_script_file.rstrip('.sh') + '.bat'
                if bootstrap_script_file is not None:
                    bootstrap_script_dir = os.path.join(unzip_package_path, "fedml",
                                                        os.path.dirname(bootstrap_script_file))
                    bootstrap_script_path = os.path.join(
                        bootstrap_script_dir, bootstrap_script_dir, os.path.basename(bootstrap_script_file)
                    )

        bootstrap_cmd_list = list()
        if bootstrap_script_path:
            logging.info("Bootstrap commands are being generated...")
            bootstrap_cmd_list = JobRunnerUtils.generate_bootstrap_commands(bootstrap_script_path=bootstrap_script_path,
                                                                            bootstrap_script_dir=bootstrap_script_dir,
                                                                            bootstrap_script_file=bootstrap_script_file)
            logging.info(f"Generated following Bootstrap commands: {bootstrap_cmd_list}")

        if not containerize:
            if len(bootstrap_cmd_list) and not (job_type == Constants.JOB_TASK_TYPE_DEPLOY or
                                                job_type == Constants.JOB_TASK_TYPE_SERVE):
                bootstrapping_successful = self.run_bootstrap_script(bootstrap_cmd_list=bootstrap_cmd_list,
                                                                     bootstrap_script_file=bootstrap_script_file)

                if not bootstrapping_successful:
                    logging.info("failed to update local fedml config.")
                    self.check_runner_stop_event()
                    # Send failed msg when exceptions.
                    self.cleanup_run_when_starting_failed(status=ClientConstants.MSG_MLOPS_CLIENT_STATUS_EXCEPTION)
                    raise Exception(f"Failed to execute following bootstrap commands: {bootstrap_cmd_list}")

                logging.info("cleanup the previous learning process and bootstrap process...")
                ClientConstants.cleanup_learning_process(self.request_json["runId"])
                ClientConstants.cleanup_bootstrap_process(self.request_json["runId"])

        executable_interpreter = ClientConstants.CLIENT_SHELL_PS \
            if platform.system() == ClientConstants.PLATFORM_WINDOWS else ClientConstants.CLIENT_SHELL_BASH

        if job_yaml_default_none is None:
            # Generate the job executing commands for previous federated learning (Compatibility)
            python_program = get_python_program()
            logging.info("Run the client: {} {} --cf {} --rank {} --role client".format(
                python_program, entry_file_full_path, conf_file_full_path, str(dynamic_args_config.get("rank", 1))))
            rank = str(dynamic_args_config.get("rank", 1))
            entry_command = f"{python_program} {entry_file_full_path} --cf " \
                            f"{conf_file_full_path} --rank {rank} --role client"
            shell_cmd_list = [entry_command]

            # Run the job executing commands for previous federated learning (Compatibility)
            process, error_list = ClientConstants.execute_commands_with_live_logs(
                shell_cmd_list, callback=self.callback_start_fl_job, should_write_log_file=False)
            is_launch_task = False
        else:
            self.check_runner_stop_event()

            self.mlops_metrics.report_client_id_status(
                self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_RUNNING, run_id=self.run_id)

            # Generate the job executing commands
            job_executing_commands = JobRunnerUtils.generate_job_execute_commands(
                self.run_id, self.edge_id, self.version,
                self.package_type, executable_interpreter, entry_file_full_path,
                conf_file_object, entry_args, assigned_gpu_ids,
                job_api_key, client_rank, scheduler_match_info=scheduler_match_info,
                cuda_visible_gpu_ids_str=self.cuda_visible_gpu_ids_str)

            if containerize is not None and containerize is True:
                docker_args = fedml_config_object.get("docker", {})
                docker_args = JobRunnerUtils.create_instance_from_dict(DockerArgs, docker_args)
                try:
                    job_executing_commands = JobRunnerUtils.generate_launch_docker_command(docker_args=docker_args,
                                                                                           run_id=self.run_id,
                                                                                           edge_id=self.edge_id,
                                                                                           unzip_package_path=unzip_package_path,
                                                                                           executable_interpreter=executable_interpreter,
                                                                                           entry_file_full_path=entry_file_full_path,
                                                                                           bootstrap_cmd_list=bootstrap_cmd_list,
                                                                                           cuda_visible_gpu_ids_str=self.cuda_visible_gpu_ids_str,
                                                                                           image_pull_policy=image_pull_policy)
                except Exception as e:
                    logging.error(f"Error occurred while generating containerized launch commands. "
                                  f"Exception: {e}, Traceback: {traceback.format_exc()}")
                    return None, None, None

                if not job_executing_commands:
                    raise Exception("Failed to generate docker execution command")

            # Run the job executing commands
            logging.info(f"Run the client job with job id {self.run_id}, device id {self.edge_id}.")
            process, error_list = ClientConstants.execute_commands_with_live_logs(
                job_executing_commands, callback=self.start_job_perf, error_processor=self.job_error_processor,
                should_write_log_file=False if job_type == Constants.JOB_TASK_TYPE_FEDERATE else True)
            is_launch_task = False if job_type == Constants.JOB_TASK_TYPE_FEDERATE else True

        return process, is_launch_task, error_list

    def callback_start_fl_job(self, job_pid):
        ClientConstants.save_learning_process(self.run_id, job_pid)
        self.mlops_metrics.report_sys_perf(
            self.args, self.agent_config["mqtt_config"], job_process_id=job_pid)

    def start_job_perf(self, job_pid):
        ClientConstants.save_learning_process(self.run_id, job_pid)
        self.mlops_metrics.report_job_perf(self.args, self.agent_config["mqtt_config"], job_pid)

    def job_error_processor(self, error_list):
        self.check_runner_stop_event()

        error_str = "\n".join(error_list)
        error_message = f"Error occurred when running the job... {error_str}"
        logging.error(error_message)
        raise Exception(error_message)

    def reset_devices_status(self, edge_id, status, should_send_client_id_status=True):
        self.mlops_metrics.run_id = self.run_id
        self.mlops_metrics.edge_id = edge_id

        if should_send_client_id_status:
            if status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED or \
                    status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED or \
                    status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_EXCEPTION:
                self.mlops_metrics.report_client_id_status(
                    edge_id, status, server_id=self.server_id, run_id=self.run_id)

    def sync_run_stop_status(self, run_status=ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED):
        try:
            if self.run_process_event is not None:
                self.run_process_event.set()

            self.mlops_metrics.report_client_id_status(
                self.edge_id, run_status, server_id=self.server_id, run_id=self.run_id)
        except Exception as e:
            logging.error(f"Failed to sync run stop status with Exception {e}. Traceback: {traceback.format_exc()}")
            pass

    def cleanup_run_when_starting_failed(
            self, status=ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED, should_send_client_id_status=True):
        # logging.error("Cleanup run successfully when starting failed.")

        self.reset_devices_status(
            self.edge_id, status, should_send_client_id_status=should_send_client_id_status)

        time.sleep(2)

        try:
            self.mlops_metrics.stop_sys_perf()
        except Exception as ex:
            logging.error(f"Failed to stop sys perf with Exception {ex}. Traceback: {traceback.format_exc()}")
            pass

        time.sleep(1)

        try:
            ClientConstants.cleanup_learning_process(self.run_id)
            ClientConstants.cleanup_bootstrap_process(self.run_id)
            ClientConstants.cleanup_run_process(self.run_id)
        except Exception as e:
            logging.error(
                f"Failed to cleanup run when starting failed with Exception {e}. Traceback: {traceback.format_exc()}")
            pass

    def cleanup_run_when_finished(self):
        # logging.info("Cleanup run successfully when finished.")

        self.reset_devices_status(self.edge_id,
                                  ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED,
                                  should_send_client_id_status=False)

        time.sleep(2)

        try:
            self.mlops_metrics.stop_sys_perf()
        except Exception as ex:
            logging.error(f"Failed to stop sys perf with Exception {ex}. Traceback: {traceback.format_exc()}")
            pass

        time.sleep(1)

        try:
            ClientConstants.cleanup_learning_process(self.run_id)
            ClientConstants.cleanup_bootstrap_process(self.run_id)
            ClientConstants.cleanup_run_process(self.run_id)
        except Exception as e:
            logging.error(
                f"Failed to cleanup run when finished with Exception {e}. Traceback: {traceback.format_exc()}")
            pass

    def setup_message_center(self):
        if self.message_center is not None:
            return

        self.message_center = FedMLMessageCenter(agent_config=self.agent_config)
        self.message_center.start_sender()

        if self.mlops_metrics is None:
            self.mlops_metrics = MLOpsMetrics()
        self.mlops_metrics.set_messenger(self.message_center)
        self.mlops_metrics.run_id = self.run_id

    def rebuild_message_center(self, message_center_queue):
        self.message_center = FedMLMessageCenter(message_queue=message_center_queue)

        if self.mlops_metrics is None:
            self.mlops_metrics = MLOpsMetrics()
        self.mlops_metrics.set_messenger(self.message_center)
        self.mlops_metrics.run_id = self.run_id

    def release_message_center(self):
        try:
            if self.message_center is not None:
                self.message_center.stop()
                self.message_center = None

        except Exception as e:
            logging.error(
                f"Failed to release client mqtt manager with Exception {e}. Traceback: {traceback.format_exc()}")
            pass

    def ota_upgrade(self, payload, request_json):
        run_id = request_json["runId"]
        force_ota = False
        ota_version = None

        try:
            run_config = request_json.get("run_config", None)
            parameters = run_config.get("parameters", None)
            common_args = parameters.get("common_args", None)
            force_ota = common_args.get("force_ota", False) if common_args is not None else False
            ota_version = common_args.get("ota_version", None) if common_args is not None else None
        except Exception as e:
            logging.error(
                f"Failed to get ota upgrade parameters with Exception {e}. Traceback: {traceback.format_exc()}")
            pass

        if force_ota and ota_version is not None:
            should_upgrade = True if ota_version != fedml.__version__ else False
            upgrade_version = ota_version
        else:
            try:
                fedml_is_latest_version, local_ver, remote_ver = sys_utils.check_fedml_is_latest_version(self.version)
            except Exception as e:
                logging.error(f"Failed to check fedml version with Exception {e}. Traceback: {traceback.format_exc()}")
                return

            should_upgrade = False if fedml_is_latest_version else True
            upgrade_version = remote_ver

        if should_upgrade:
            FedMLClientDataInterface.get_instance(). \
                save_started_job(run_id, self.edge_id, time.time(),
                                 ClientConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING,
                                 ClientConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING,
                                 payload)
            self.mlops_metrics.report_client_id_status(
                self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING, run_id=run_id)

            logging.info(f"Upgrade to version {upgrade_version} ...")

            sys_utils.do_upgrade(self.version, upgrade_version)
            raise Exception("Restarting after upgraded...")

    def callback_start_train(self, topic, payload):
        # Get training params

        request_json = json.loads(payload)
        is_retain = request_json.get("is_retain", False)
        if is_retain:
            return
        run_id = request_json["runId"]

        # Start log processor for current run
        train_edge_id = str(topic).split("/")[-2]
        self.args.run_id = run_id
        self.args.edge_id = train_edge_id
        MLOpsRuntimeLog.get_instance(self.args).init_logs(log_level=logging.INFO)
        MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(
            run_id, train_edge_id, log_source=SchedulerConstants.get_log_source(request_json))
        logging.info("start the log processor")

        try:
            MLOpsConfigs.fetch_all_configs()
        except Exception as e:
            logging.error(f"Failed to fetch all configs with Exception {e}. Traceback: {traceback.format_exc()}")
            pass

        if not FedMLClientDataInterface.get_instance().get_agent_status():
            request_json = json.loads(payload)
            run_id = request_json["runId"]
            logging.error(
                "FedMLDebug - Receive: topic ({}), payload ({}), but the client agent is disabled. {}".format(
                    topic, payload, traceback.format_exc()
                )
            )
            # Send failed msg when exceptions.
            self.mlops_metrics.report_client_id_status(
                train_edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_EXCEPTION, run_id=run_id,
                msg=f"the client agent {train_edge_id} is disabled")
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, train_edge_id)
            return

        logging.info(
            f"FedMLDebug - Receive: topic ({topic}), payload ({payload})"
        )

        # Terminate previous process about starting or stopping run command
        logging.info("cleanup and save runner information")
        server_agent_id = request_json["cloud_agent_id"]
        ClientConstants.save_runner_infos(self.args.device_id + "." + self.args.os_name, train_edge_id, run_id=run_id)

        # OTA upgrade
        # self.ota_upgrade(payload, request_json)

        # Occupy GPUs
        scheduler_match_info = request_json.get("scheduler_match_info", {})
        matched_gpu_num = scheduler_match_info.get("matched_gpu_num", 0)
        model_master_device_id = scheduler_match_info.get("model_master_device_id", None)
        model_slave_device_id = scheduler_match_info.get("model_slave_device_id", None)
        model_slave_device_id_list = scheduler_match_info.get("model_slave_device_id_list", None)
        run_config = request_json.get("run_config", {})
        run_params = run_config.get("parameters", {})
        serving_args = run_params.get("serving_args", {})
        endpoint_id = serving_args.get("endpoint_id", None)
        cuda_visible_gpu_ids_str = JobRunnerUtils.get_instance().occupy_gpu_ids(
            run_id, matched_gpu_num, train_edge_id, inner_id=endpoint_id,
            model_master_device_id=model_master_device_id,
            model_slave_device_id=model_slave_device_id)
        logging.info(
            f"Run started, available gpu ids: {JobRunnerUtils.get_instance().get_available_gpu_id_list(train_edge_id)}")

        # Start server with multiprocessing mode
        self.request_json = request_json
        run_id_str = str(run_id)
        self.running_request_json[run_id_str] = request_json
        client_runner = FedMLClientRunner(
            self.args, edge_id=train_edge_id, request_json=request_json, agent_config=self.agent_config, run_id=run_id,
            cuda_visible_gpu_ids_str=cuda_visible_gpu_ids_str
        )
        client_runner.start_request_json = payload
        self.run_process_event_map[run_id_str] = multiprocessing.Event()
        self.run_process_event_map[run_id_str].clear()
        client_runner.run_process_event = self.run_process_event_map[run_id_str]
        self.run_process_completed_event_map[run_id_str] = multiprocessing.Event()
        self.run_process_completed_event_map[run_id_str].clear()
        client_runner.run_process_completed_event = self.run_process_completed_event_map[run_id_str]
        client_runner.server_id = request_json.get("server_id", "0")
        logging.info("start the runner process.")
        self.run_process_map[run_id_str] = Process(target=client_runner.run, args=(
            self.run_process_event_map[run_id_str], self.run_process_completed_event_map[run_id_str],
            self.message_center.get_message_queue()))
        self.run_process_map[run_id_str].start()
        ClientConstants.save_run_process(run_id, self.run_process_map[run_id_str].pid)

    def callback_stop_train(self, topic, payload):
        # logging.info("callback_stop_train: topic = %s, payload = %s" % (topic, payload))
        # logging.info(
        #     f"FedMLDebug - Receive: topic ({topic}), payload ({payload})"
        # )

        train_edge_id = str(topic).split("/")[-2]
        request_json = json.loads(payload)
        is_retain = request_json.get("is_retain", False)
        if is_retain:
            return
        run_id = request_json.get("runId", None)
        if run_id is None:
            run_id = request_json.get("id", None)
        run_status = request_json.get("run_status", ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED)

        # logging.info("Stop run with multiprocessing...")

        # Stop client with multiprocessing mode
        run_id_str = str(run_id)
        client_runner = FedMLClientRunner(
            self.args, edge_id=train_edge_id, request_json=request_json, agent_config=self.agent_config, run_id=run_id
        )
        self.cleanup_containers_and_release_gpus(run_id, train_edge_id)
        client_runner.run_process_event = self.run_process_event_map.get(run_id_str, None)
        client_runner.run_process = self.run_process_map.get(run_id_str, None)
        client_runner.message_center = self.message_center
        client_runner.mlops_metrics = self.mlops_metrics
        client_runner.sync_run_stop_status(run_status=run_status)

    def cleanup_containers_and_release_gpus(self, run_id, edge_id):
        job_type = JobRunnerUtils.get_job_type_from_run_id(run_id)

        if not job_type:
            logging.info(f"Failed to get job type from run id {run_id}. This is not an error as it would usually "
                         f"happen when the job is not found in the database because job is already finished and "
                         f"cleaned up. Exiting cleanup_containers_and_release_gpus.")
            return

        # Check if the job type is not "serve" or "deploy"
        if not (job_type == SchedulerConstants.JOB_TASK_TYPE_SERVE or
                job_type == SchedulerConstants.JOB_TASK_TYPE_DEPLOY):

            # Terminate the run docker container if exists
            container_name = JobRunnerUtils.get_run_container_name(run_id)
            docker_client = JobRunnerUtils.get_docker_client(DockerArgs())
            logging.info(f"Terminating the run docker container {container_name} if exists...")
            try:
                JobRunnerUtils.remove_run_container_if_exists(container_name, docker_client)
            except Exception as e:
                logging.error(f"Exception {e} occurred when terminating docker container. "
                              f"Traceback: {traceback.format_exc()}")

            # Release the GPU ids and update the GPU availability in the persistent store
            JobRunnerUtils.get_instance().release_gpu_ids(run_id, edge_id)

            # Send mqtt message reporting the new gpu availability to the backend
            MLOpsDevicePerfStats.report_gpu_device_info(self.edge_id, mqtt_mgr=self.mqtt_mgr)

    def cleanup_client_with_status(self):
        if self.device_status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED:
            # logging.info("received to finished status.")
            self.cleanup_run_when_finished()
        elif self.device_status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED:
            # logging.error("received to failed status from the server agent")
            self.cleanup_run_when_starting_failed(should_send_client_id_status=False)
        elif self.device_status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED:
            # logging.error("received to failed status from the server agent")
            self.cleanup_run_when_starting_failed(status=self.device_status, should_send_client_id_status=False)

    def callback_runner_id_status(self, topic, payload):
        # logging.info("callback_runner_id_status: topic = %s, payload = %s" % (topic, payload))
        # logging.info(f"FedMLDebug - Receive: topic ({topic}), payload ({payload})")
        request_json = json.loads(payload)
        is_retain = request_json.get("is_retain", False)
        if is_retain:
            return
        run_id = request_json["run_id"]
        edge_id = str(topic).split("/")[-2].split('_')[-1]
        status = request_json["status"]
        run_id_str = str(run_id)

        self.save_training_status(
            edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED
            if status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_EXCEPTION else status)

        if status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED or \
                status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED or \
                status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED:
            completed_event = self.run_process_completed_event_map.get(run_id_str, None)
            if completed_event is not None:
                completed_event.set()

            # Stop client with multiprocessing mode
            client_runner = FedMLClientRunner(
                self.args,
                edge_id=edge_id,
                request_json=request_json,
                agent_config=self.agent_config,
                run_id=run_id,
            )
            client_runner.device_status = status
            client_runner.message_center = self.message_center
            client_runner.mlops_metrics = self.mlops_metrics
            client_runner.cleanup_client_with_status()

            running_json = self.running_request_json.get(run_id_str)
            if running_json is None:
                try:
                    current_job = FedMLClientDataInterface.get_instance().get_job_by_id(run_id)
                    running_json = json.loads(current_job.running_json)
                except Exception as e:
                    logging.error(f"Failed to get running json with Exception {e}. Traceback: {traceback.format_exc()}")

            if running_json is not None:
                job_type = JobRunnerUtils.parse_job_type(running_json)
                if not SchedulerConstants.is_deploy_job(job_type):
                    logging.info(f"[run/device][{run_id}/{edge_id}] Release gpu resource when run ended.")
                    self.cleanup_containers_and_release_gpus(run_id, edge_id)

            run_process = self.run_process_map.get(run_id_str, None)
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

                self.run_process_map.pop(run_id_str)

            # Stop log processor for current run
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, edge_id)

    def callback_report_current_status(self, topic, payload):
        logging.info(
            f"FedMLDebug - Receive: topic ({topic}), payload ({payload})"
        )

        self.send_agent_active_msg()
        if self.general_edge_id is not None:
            self.send_agent_active_msg(self.general_edge_id)

    @staticmethod
    def process_ota_upgrade_msg():
        os.system("pip install -U fedml")

    @staticmethod
    def callback_client_ota_msg(topic, payload):
        logging.info(
            f"FedMLDebug - Receive: topic ({topic}), payload ({payload})"
        )

        request_json = json.loads(payload)
        cmd = request_json["cmd"]

        if cmd == ClientConstants.FEDML_OTA_CMD_UPGRADE:
            FedMLClientRunner.process_ota_upgrade_msg()
            # Process(target=FedMLClientRunner.process_ota_upgrade_msg).start()
            raise Exception("After upgraded, restart runner...")
        elif cmd == ClientConstants.FEDML_OTA_CMD_RESTART:
            raise Exception("Restart runner...")

    def get_all_run_process_list_map(self):
        run_process_dict = dict()
        for run_id_str, process in self.run_process_map.items():
            cur_run_process_list = ClientConstants.get_learning_process_list(run_id_str)
            run_process_dict[run_id_str] = cur_run_process_list

        return run_process_dict

    def response_device_info_to_mlops(self, topic, payload):
        payload_json = json.loads(payload)
        server_id = payload_json.get("server_id", 0)
        run_id = payload_json.get("run_id", 0)
        listen_edge_id = str(topic).split("/")[-1]
        context = payload_json.get("context", None)
        need_gpu_info = payload_json.get("need_gpu_info", False)
        need_running_process_list = payload_json.get("need_running_process_list", False)
        response_topic = f"deploy/slave_agent/mlops/response_device_info"
        if self.mlops_metrics is not None and self.model_device_client_edge_id_list is not None and \
                self.model_device_server_id is not None:
            if not need_gpu_info:
                device_info_json = {
                    "edge_id": listen_edge_id,
                    "fedml_version": fedml.__version__,
                    "user_id": self.args.user
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
                    "user_id": self.args.user
                }
            if need_running_process_list:
                device_info_json["run_process_list_map"] = self.get_all_run_process_list_map()
            salve_device_ids = list()
            for model_client_edge_id in self.model_device_client_edge_id_list:
                salve_device_ids.append(model_client_edge_id)
            response_payload = {"slave_device_id": self.model_device_client_edge_id_list[0],
                                "slave_device_id_list": salve_device_ids,
                                "master_device_id": self.model_device_server_id,
                                "run_id": run_id, "edge_id": listen_edge_id,
                                "edge_info": device_info_json}
            if context is not None:
                response_payload["context"] = context
            self.message_center.send_message(response_topic, json.dumps(response_payload), run_id=run_id)
    
    def callback_report_device_info(self, topic, payload):
        payload_json = json.loads(payload)
        server_id = payload_json.get("server_id", 0)
        run_id = payload_json.get("run_id", 0)
        listen_edge_id = str(topic).split("/")[-1]
        context = payload_json.get("context", None)
        need_gpu_info = payload_json.get("need_gpu_info", False)
        need_running_process_list = payload_json.get("need_running_process_list", False)
        response_topic = f"client/server/response_device_info/{server_id}"
        if self.mlops_metrics is not None and self.model_device_client_edge_id_list is not None and \
                self.model_device_server_id is not None:
            if not need_gpu_info:
                device_info_json = {
                    "edge_id": listen_edge_id,
                    "fedml_version": fedml.__version__,
                    "user_id": self.args.user
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
                    "user_id": self.args.user
                }
            if need_running_process_list:
                device_info_json["run_process_list_map"] = self.get_all_run_process_list_map()
            salve_device_ids = list()
            for model_client_edge_id in self.model_device_client_edge_id_list:
                salve_device_ids.append(model_client_edge_id)
            response_payload = {"slave_device_id": self.model_device_client_edge_id_list[0],
                                "slave_device_id_list": salve_device_ids,
                                "master_device_id": self.model_device_server_id,
                                "run_id": run_id, "edge_id": listen_edge_id,
                                "edge_info": device_info_json}
            if context is not None:
                response_payload["context"] = context
            self.message_center.send_message(response_topic, json.dumps(response_payload), run_id=run_id)

    def callback_client_logout(self, topic, payload):
        payload_json = json.loads(payload)
        secret = payload_json.get("auth", None)
        if secret is None or str(secret) != "246b1be6-0eeb-4b17-b118-7d74de1975d4":
            return
        logging.info("Received the logout request.")
        if self.run_process_event is not None:
            self.run_process_event.set()
        if self.run_process_completed_event is not None:
            self.run_process_completed_event.set()
        self.disable_client_login = True
        time.sleep(3)
        os.system("fedml logout")

    def save_training_status(self, edge_id, training_status):
        self.current_training_status = training_status
        ClientConstants.save_training_infos(edge_id, training_status)

    @staticmethod
    def get_gpu_machine_id():
        gpu_list = sys_utils.get_gpu_list()
        gpu_uuids = ""
        if len(gpu_list) > 0:
            for gpu in gpu_list:
                gpu_uuids += gpu.get("uuid", "")
        else:
            gpu_uuids = str(uuid.uuid4())
        device_id_combination = \
            f"{FedMLClientRunner.get_machine_id()}-{hex(uuid.getnode())}-{gpu_uuids}"
        device_id = security_utils.get_content_hash(device_id_combination)
        return device_id

    @staticmethod
    def get_device_id(use_machine_id=False):
        device_file_path = os.path.join(ClientConstants.get_data_dir(),
                                        ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME)
        file_for_device_id = os.path.join(device_file_path, "devices.id")
        if not os.path.exists(device_file_path):
            os.makedirs(device_file_path, exist_ok=True)
        elif os.path.exists(file_for_device_id):
            with open(file_for_device_id, 'r', encoding='utf-8') as f:
                device_id_from_file = f.readline()
                if device_id_from_file is not None and device_id_from_file != "":
                    return device_id_from_file

        if platform.system() == "Darwin":
            cmd_get_serial_num = "system_profiler SPHardwareDataType | grep Serial | awk '{gsub(/ /,\"\")}{print}' " \
                                 "|awk -F':' '{print $2}' "
            device_id = os.popen(cmd_get_serial_num).read()
            device_id = device_id.replace('\n', '').replace(' ', '')
            if device_id is None or device_id == "":
                if not use_machine_id:
                    device_id = hex(uuid.getnode())
                else:
                    device_id = FedMLClientRunner.get_gpu_machine_id()
            else:
                device_id = "0x" + device_id
        else:
            if "nt" in os.name:

                def get_uuid():
                    guid = ""
                    try:
                        cmd = "wmic csproduct get uuid"
                        guid = str(subprocess.check_output(cmd))
                        pos1 = guid.find("\\n") + 2
                        guid = guid[pos1:-15]
                    except Exception as ex:
                        logging.error(f"Failed to get uuid with Exception {ex}. Traceback: {traceback.format_exc()}")
                        pass
                    return str(guid)

                device_id = str(get_uuid())
                logging.info(device_id)
            elif "posix" in os.name:
                device_id = sys_utils.get_device_id_in_docker()
                if device_id is None:
                    if not use_machine_id:
                        device_id = hex(uuid.getnode())
                    else:
                        device_id = device_id = FedMLClientRunner.get_gpu_machine_id()
            else:
                device_id = sys_utils.run_subprocess_open(
                    "hal-get-property --udi /org/freedesktop/Hal/devices/computer --key system.hardware.uuid".split()
                )
                device_id = hex(device_id)

        if device_id is not None and device_id != "":
            with open(file_for_device_id, 'w', encoding='utf-8') as f:
                f.write(device_id)
        else:
            device_id = hex(uuid.uuid4())
            with open(file_for_device_id, 'w', encoding='utf-8') as f:
                f.write(device_id)

        return device_id

    @staticmethod
    def get_machine_id():
        try:
            import machineid
            return machineid.id().replace('\n', '').replace('\r\n', '').strip()
        except Exception as e:
            logging.error(f"Failed to get machine id with Exception {e}. Traceback: {traceback.format_exc()}")
            return hex(uuid.getnode())

    @staticmethod
    def bind_account_and_device_id(url, account_id, device_id, os_name, api_key="", role="client"):
        ip = requests.get('https://checkip.amazonaws.com').text.strip()
        fedml_ver, exec_path, os_ver, cpu_info, python_ver, torch_ver, mpi_installed, \
            cpu_usage, available_mem, total_mem, gpu_info, gpu_available_mem, gpu_total_mem, \
            gpu_count, gpu_vendor, cpu_count, gpu_device_name = get_sys_runner_info()
        host_name = sys_utils.get_host_name()
        json_params = {
            "accountid": account_id,
            "deviceid": device_id,
            "type": os_name,
            "state": ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE,
            "status": ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE,
            "processor": cpu_info,
            "core_type": cpu_info,
            "network": "",
            "role": role,
            "os_ver": os_ver,
            "memory": total_mem,
            "ip": ip,
            "api_key": api_key,
            "extra_infos": {"fedml_ver": fedml_ver, "exec_path": exec_path, "os_ver": os_ver,
                            "cpu_info": cpu_info, "python_ver": python_ver, "torch_ver": torch_ver,
                            "mpi_installed": mpi_installed, "cpu_usage": cpu_usage,
                            "available_mem": available_mem, "total_mem": total_mem,
                            "cpu_count": cpu_count, "gpu_count": 0, "host_name": host_name}
        }
        if gpu_count > 0:
            if gpu_total_mem is not None:
                json_params["gpu"] = gpu_info if gpu_info is not None else "" + ", Total GPU Memory: " + gpu_total_mem
            else:
                json_params["gpu"] = gpu_info if gpu_info is not None else ""
            json_params["extra_infos"]["gpu_info"] = gpu_info if gpu_info is not None else ""
            if gpu_available_mem is not None:
                json_params["extra_infos"]["gpu_available_mem"] = gpu_available_mem
            if gpu_total_mem is not None:
                json_params["extra_infos"]["gpu_total_mem"] = gpu_total_mem

            json_params["extra_infos"]["gpu_count"] = gpu_count
            json_params["extra_infos"]["gpu_vendor"] = gpu_vendor
            json_params["extra_infos"]["gpu_device_name"] = gpu_device_name

            gpu_available_id_list = sys_utils.get_available_gpu_id_list(limit=gpu_count)
            gpu_available_count = len(gpu_available_id_list) if gpu_available_id_list is not None else 0
            gpu_list = sys_utils.get_gpu_list()
            json_params["extra_infos"]["gpu_available_count"] = gpu_available_count
            json_params["extra_infos"]["gpu_available_id_list"] = gpu_available_id_list
            json_params["extra_infos"]["gpu_list"] = gpu_list
        else:
            json_params["gpu"] = "None"
            json_params["extra_infos"]["gpu_available_count"] = 0
            json_params["extra_infos"]["gpu_available_id_list"] = []
            json_params["extra_infos"]["gpu_list"] = []

        _, cert_path = MLOpsConfigs.get_request_params()
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    url, json=json_params, verify=True,
                    headers={"content-type": "application/json", "Connection": "close"}
                )
            except requests.exceptions.SSLError as err:
                logging.error(
                    f"Failed to bind account and device id with error: {err}, traceback: {traceback.format_exc()}")
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    url, json=json_params, verify=True,
                    headers={"content-type": "application/json", "Connection": "close"}
                )
        else:
            response = requests.post(url, json=json_params, headers={"Connection": "close"})
        edge_id, user_name, extra_url, general_edge_id = -1, None, None, None
        if response.status_code != 200:
            print(f"Binding to MLOps with response.status_code = {response.status_code}, "
                  f"response.content: {response.content}")
            pass
        else:
            # print("url = {}, response = {}".format(url, response))
            status_code = response.json().get("code")
            if status_code == "SUCCESS":
                edge_id = response.json().get("data").get("id")
                user_name = response.json().get("data").get("userName", None)
                extra_url = response.json().get("data").get("url", None)
                general_edge_id = response.json().get("data").get("general_edge_id", None)
                if edge_id is None or edge_id <= 0:
                    print(f"Binding to MLOps with response.status_code = {response.status_code}, "
                          f"response.content: {response.content}")
            else:
                if status_code == SchedulerConstants.BINDING_ACCOUNT_NOT_EXIST_ERROR:
                    raise SystemExit(SchedulerConstants.BINDING_ACCOUNT_NOT_EXIST_ERROR)
                print(f"Binding to MLOps with response.status_code = {response.status_code}, "
                      f"response.content: {response.content}")
                return -1, None, None, None
        return edge_id, user_name, extra_url, general_edge_id

    def fetch_configs(self):
        return MLOpsConfigs.fetch_all_configs()

    def send_agent_active_msg(self, edge_id):
        active_topic = "flclient_agent/active"
        status = MLOpsStatus.get_instance().get_client_agent_status(edge_id)
        if (
                status is not None
                and status != ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE
                and status != ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE
        ):
            return

        try:
            current_job = FedMLClientDataInterface.get_instance().get_job_by_id(self.run_id)
        except Exception as e:
            logging.error(f"Failed to get current job with Exception {e}. Traceback: {traceback.format_exc()}")
            current_job = None
        if current_job is None:
            if status is not None and status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE:
                status = ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE
            else:
                return
        else:
            status = ClientConstants.get_device_state_from_run_edge_state(current_job.status)
        active_msg = {"ID": edge_id, "status": status}
        MLOpsStatus.get_instance().set_client_agent_status(edge_id, status)
        self.mqtt_mgr.send_message_json(active_topic, json.dumps(active_msg))
        logging.info(f"Send agent active msg {active_msg}")

    def recover_start_train_msg_after_upgrading(self):
        try:
            current_job = FedMLClientDataInterface.get_instance().get_current_job()
            if current_job is not None and \
                    current_job.status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING:
                logging.info("start training after upgrading.")
                topic_start_train = "flserver_agent/" + str(self.edge_id) + "/start_train"
                self.callback_start_train(topic_start_train, current_job.running_json)
        except Exception as e:
            logging.error(f"recover starting train message after upgrading failed with exception {e}, "
                          f"Traceback {traceback.format_exc()}")

    def on_agent_mqtt_connected(self, mqtt_client_object):
        # The MQTT message topic format is as follows: <sender>/<receiver>/<action>

        # Setup MQTT message listener for starting training
        topic_start_train = "flserver_agent/" + str(self.edge_id) + "/start_train"
        self.add_message_listener(topic_start_train, self.callback_start_train)
        self.mqtt_mgr.add_message_listener(topic_start_train, self.listener_message_dispatch_center)

        # Setup MQTT message listener for stopping training
        topic_stop_train = "flserver_agent/" + str(self.edge_id) + "/stop_train"
        self.add_message_listener(topic_stop_train, self.callback_stop_train)
        self.mqtt_mgr.add_message_listener(topic_stop_train, self.listener_message_dispatch_center)


        # Setup MQTT message listener for client status switching
        topic_client_status = "fl_client/flclient_agent_" + str(self.edge_id) + "/status"
        self.add_message_listener(topic_client_status, self.callback_runner_id_status)
        self.mqtt_mgr.add_message_listener(topic_client_status, self.listener_message_dispatch_center)

        # Setup MQTT message listener to report current device status.
        topic_report_status = "mlops/report_device_status"
        self.add_message_listener(topic_report_status, self.callback_report_current_status)
        self.mqtt_mgr.add_message_listener(topic_report_status, self.listener_message_dispatch_center)

        # Setup MQTT message listener to OTA messages from the MLOps.
        topic_ota_msg = "mlops/flclient_agent_" + str(self.edge_id) + "/ota"
        self.add_message_listener(topic_ota_msg, self.callback_client_ota_msg)
        self.mqtt_mgr.add_message_listener(topic_ota_msg, self.listener_message_dispatch_center)

        # Setup MQTT message listener to OTA messages from the MLOps.
        topic_request_device_info = "server/client/request_device_info/" + str(self.edge_id)
        self.add_message_listener(topic_request_device_info, self.callback_report_device_info)
        self.mqtt_mgr.add_message_listener(topic_request_device_info, self.listener_message_dispatch_center)

        topic_request_edge_device_info_from_mlops = f"deploy/mlops/slave_agent/request_device_info/{self.edge_id}"
        self.add_message_listener(topic_request_edge_device_info_from_mlops, self.response_device_info_to_mlops)
        self.mqtt_mgr.add_message_listener(topic_request_edge_device_info_from_mlops, self.listener_message_dispatch_center)

        topic_request_deploy_master_device_info_from_mlops = None
        if self.model_device_server_id is not None:
            topic_request_deploy_master_device_info_from_mlops = f"deploy/mlops/master_agent/request_device_info/{self.model_device_server_id}"
            self.add_message_listener(topic_request_deploy_master_device_info_from_mlops, self.response_device_info_to_mlops)
            self.mqtt_mgr.add_message_listener(topic_request_deploy_master_device_info_from_mlops, self.listener_message_dispatch_center)

        topic_request_deploy_slave_device_info_from_mlops = None
        if self.model_device_client_edge_id_list is not None and len(self.model_device_client_edge_id_list) > 0:
            topic_request_deploy_slave_device_info_from_mlops = f"deploy/mlops/slave_agent/request_device_info/{self.model_device_client_edge_id_list[0]}"
            self.add_message_listener(topic_request_deploy_slave_device_info_from_mlops, self.response_device_info_to_mlops)
            self.mqtt_mgr.add_message_listener(topic_request_deploy_slave_device_info_from_mlops, self.listener_message_dispatch_center)
        
        # Setup MQTT message listener to logout from MLOps.
        topic_client_logout = "mlops/client/logout/" + str(self.edge_id)
        self.add_message_listener(topic_client_logout, self.callback_client_logout)
        self.mqtt_mgr.add_message_listener(topic_client_logout, self.listener_message_dispatch_center)

        # Subscribe topics for starting train, stopping train and fetching client status.
        mqtt_client_object.subscribe(topic_start_train, qos=2)
        mqtt_client_object.subscribe(topic_stop_train, qos=2)
        mqtt_client_object.subscribe(topic_client_status, qos=2)
        mqtt_client_object.subscribe(topic_report_status, qos=2)
        mqtt_client_object.subscribe(topic_ota_msg, qos=2)
        mqtt_client_object.subscribe(topic_request_device_info, qos=2)
        mqtt_client_object.subscribe(topic_request_edge_device_info_from_mlops, qos=2)
        if topic_request_deploy_master_device_info_from_mlops is not None:
            mqtt_client_object.subscribe(topic_request_deploy_master_device_info_from_mlops, qos=2)
        if topic_request_deploy_slave_device_info_from_mlops is not None:
            mqtt_client_object.subscribe(topic_request_deploy_slave_device_info_from_mlops, qos=2)
        mqtt_client_object.subscribe(topic_client_logout, qos=2)

        self.subscribed_topics.clear()
        self.subscribed_topics.append(topic_start_train)
        self.subscribed_topics.append(topic_stop_train)
        self.subscribed_topics.append(topic_client_status)
        self.subscribed_topics.append(topic_report_status)
        self.subscribed_topics.append(topic_ota_msg)
        self.subscribed_topics.append(topic_request_device_info)
        self.subscribed_topics.append(topic_request_edge_device_info_from_mlops)
        if topic_request_deploy_master_device_info_from_mlops is not None:
            self.subscribed_topics.append(topic_request_deploy_master_device_info_from_mlops)
        if topic_request_deploy_slave_device_info_from_mlops is not None:
            self.subscribed_topics.append(topic_request_deploy_slave_device_info_from_mlops)
        self.subscribed_topics.append(topic_client_logout)

        # Subscribe the messages for federated learning.
        self.subscribe_fl_msgs()

        # Broadcast the first active message.
        self.send_agent_active_msg(self.edge_id)
        if self.general_edge_id is not None:
            self.send_agent_active_msg(self.general_edge_id)

        # Echo results
        MLOpsRuntimeLog.get_instance(self.args).enable_show_log_to_stdout()
        worker_deploy_id_list = [modeld_device_clint.edge_id for index, modeld_device_clint in
                                 enumerate(self.model_device_client_list)]
        print("\nCongratulations, your device is connected to the FedML MLOps platform successfully!")
        print(f"Your FedML Edge ID is {str(self.edge_id)}, unique device ID is {str(self.unique_device_id)}, "
              f"master deploy ID is {str(self.model_device_server.edge_id)}, "
              f"worker deploy ID is {worker_deploy_id_list}"
              )
        if self.edge_extra_url is not None and self.edge_extra_url != "":
            print(f"You may visit the following url to fill in more information with your device.\n"
                  f"{self.edge_extra_url}")
        MLOpsRuntimeLog.get_instance(self.args).enable_show_log_to_stdout(enable=False)

        from fedml.core.mlops import sync_deploy_id
        sync_deploy_id(
            self.edge_id, self.model_device_server.edge_id, worker_deploy_id_list)

        # Start the message center for listener
        self.start_listener(sender_message_queue=self.message_center.get_message_queue(),
                            agent_config=self.agent_config)

    def subscribe_fl_msgs(self):
        if self.general_edge_id is None:
            return

        # Setup MQTT message listener for starting training
        topic_start_train = "flserver_agent/" + str(self.general_edge_id) + "/start_train"
        self.add_message_listener(topic_start_train, self.callback_start_train)
        self.mqtt_mgr.add_message_listener(topic_start_train, self.listener_message_dispatch_center)

        # Setup MQTT message listener for stopping training
        topic_stop_train = "flserver_agent/" + str(self.general_edge_id) + "/stop_train"
        self.add_message_listener(topic_stop_train, self.callback_stop_train)
        self.mqtt_mgr.add_message_listener(topic_stop_train, self.listener_message_dispatch_center)

        # Setup MQTT message listener for client status switching
        topic_client_status = "fl_client/flclient_agent_" + str(self.general_edge_id) + "/status"
        self.add_message_listener(topic_client_status, self.callback_runner_id_status)
        self.mqtt_mgr.add_message_listener(topic_client_status, self.listener_message_dispatch_center)

        # Setup MQTT message listener to OTA messages from the MLOps.
        topic_request_device_info = "server/client/request_device_info/" + str(self.general_edge_id)
        self.add_message_listener(topic_request_device_info, self.callback_report_device_info)
        self.mqtt_mgr.add_message_listener(topic_request_device_info, self.listener_message_dispatch_center)

        topic_request_device_info_from_mlops = f"deploy/mlops/client_agent/request_device_info/{self.general_edge_id}"
        self.add_message_listener(topic_request_device_info_from_mlops, self.response_device_info_to_mlops)
        self.mqtt_mgr.add_message_listener(topic_request_device_info_from_mlops, self.listener_message_dispatch_center)

        # Subscribe topics for starting train, stopping train and fetching client status.
        self.mqtt_mgr.subscribe_msg(topic_start_train)
        self.mqtt_mgr.subscribe_msg(topic_stop_train)
        self.mqtt_mgr.subscribe_msg(topic_client_status)
        self.mqtt_mgr.subscribe_msg(topic_request_device_info)
        self.mqtt_mgr.subscribe_msg(topic_request_device_info_from_mlops)

        self.subscribed_topics.append(topic_start_train)
        self.subscribed_topics.append(topic_stop_train)
        self.subscribed_topics.append(topic_client_status)
        self.subscribed_topics.append(topic_request_device_info)
        self.subscribed_topics.append(topic_request_device_info_from_mlops)

    def on_agent_mqtt_disconnected(self, mqtt_client_object):
        MLOpsStatus.get_instance().set_client_agent_status(
            self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE
        )
        pass

    def setup_agent_mqtt_connection(self, service_config):
        # Setup MQTT connection
        self.mqtt_mgr = MqttManager(
            service_config["mqtt_config"]["BROKER_HOST"],
            service_config["mqtt_config"]["BROKER_PORT"],
            service_config["mqtt_config"]["MQTT_USER"],
            service_config["mqtt_config"]["MQTT_PWD"],
            service_config["mqtt_config"]["MQTT_KEEPALIVE"],
            f"FedML_ClientAgent_Daemon_@{self.user_name}@_@{self.args.current_device_id}@_@{str(uuid.uuid4())}@",
            "flclient_agent/last_will_msg",
            json.dumps({"ID": self.edge_id, "status": ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE})
        )
        self.agent_config = service_config

        # Init local database
        FedMLClientDataInterface.get_instance().create_job_table()

        # Start the message center to process edge related messages.
        self.setup_message_center()

        # Start local API services
        client_api_cmd = "fedml.computing.scheduler.slave.client_api:api"
        client_api_pids = RunProcessUtils.get_pid_from_cmd_line(client_api_cmd)
        if client_api_pids is None or len(client_api_pids) <= 0:
            python_program = get_python_program()
            cur_dir = os.path.dirname(__file__)
            fedml_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(cur_dir)))
            self.local_api_process = ClientConstants.exec_console_with_script(
                "{} -m uvicorn {} --host 0.0.0.0 --port {} "
                "--reload --reload-delay 3 --reload-dir {} --log-level critical".format(
                    python_program, client_api_cmd, ClientConstants.LOCAL_CLIENT_API_PORT, fedml_base_dir),
                should_capture_stdout=False,
                should_capture_stderr=False
            )
            # if self.local_api_process is not None and self.local_api_process.pid is not None:
            #     print(f"Client local API process id {self.local_api_process.pid}")

        # Setup MQTT connected listener
        self.mqtt_mgr.add_connected_listener(self.on_agent_mqtt_connected)
        self.mqtt_mgr.add_disconnected_listener(self.on_agent_mqtt_disconnected)
        self.mqtt_mgr.connect()

        # Report the IDLE status to MLOps
        self.mlops_metrics.report_client_training_status(
            self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE)
        MLOpsStatus.get_instance().set_client_agent_status(self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE)

        # MLOpsRuntimeLogDaemon.get_instance(self.args).stop_all_log_processor()
        self.recover_start_train_msg_after_upgrading()

        infer_host = os.getenv("FEDML_INFER_HOST", None)
        infer_redis_addr = os.getenv("FEDML_INFER_REDIS_ADDR", None)
        infer_redis_port = os.getenv("FEDML_INFER_REDIS_PORT", None)
        infer_redis_password = os.getenv("FEDML_INFER_REDIS_PASSWORD", None)
        model_client_num = os.getenv("FEDML_MODEL_WORKER_NUM", None)
        os.environ["FEDML_CURRENT_EDGE_ID"] = str(self.edge_id)

        if not ComputeCacheManager.get_instance().set_redis_params():
            os.environ["FEDML_DISABLE_REDIS_CONNECTION"] = "1"

        if self.model_device_client_edge_id_list is None:
            self.model_device_client_edge_id_list = list()
        if self.model_device_client_list is None:
            model_client_num = 1 if model_client_num is None else int(model_client_num)
            self.model_device_client_list = list()
            for client_index in range(model_client_num):
                model_device_client = FedMLModelDeviceClientRunner(
                    self.args, f"{self.args.current_device_id}_{client_index + 1}", self.args.os_name,
                    self.args.is_from_docker, self.agent_config)
                if infer_host is not None:
                    model_device_client.infer_host = infer_host
                if infer_redis_addr is not None:
                    model_device_client.redis_addr = infer_redis_addr
                if infer_redis_port is not None:
                    model_device_client.redis_port = infer_redis_port
                if infer_redis_password is not None:
                    model_device_client.redis_password = infer_redis_password
                model_device_client.start()
                self.model_device_client_list.append(model_device_client)
                self.model_device_client_edge_id_list.append(model_device_client.get_edge_id())

        if self.model_device_server is None:
            self.model_device_server = FedMLModelDeviceServerRunner(self.args, self.args.current_device_id,
                                                                    self.args.os_name, self.args.is_from_docker,
                                                                    self.agent_config)
            if infer_host is not None:
                self.model_device_server.infer_host = infer_host
            if infer_redis_addr is not None:
                self.model_device_server.redis_addr = infer_redis_addr
            if infer_redis_port is not None:
                self.model_device_server.redis_port = infer_redis_port
            if infer_redis_password is not None:
                self.model_device_server.redis_password = infer_redis_password

            self.model_device_server.start()
            self.model_device_server_id = self.model_device_server.get_edge_id()

        JobCleanup.get_instance().sync_data_on_startup(self.edge_id)

        os.environ["FEDML_DEPLOY_MASTER_ID"] = str(self.model_device_server.get_edge_id())
        os.environ["FEDML_DEPLOY_WORKER_IDS"] = str([client.get_edge_id() for client in self.model_device_client_list])
        self.mlops_metrics.stop_device_realtime_perf()
        self.mlops_metrics.report_device_realtime_perf(self.args, service_config["mqtt_config"])

    def start_agent_mqtt_loop(self):
        # Start MQTT message loop
        try:
            self.mqtt_mgr.loop_forever()
        except Exception as e:
            logging.error(f"Errors in the MQTT loop: Exception {e}, Traceback: {traceback.format_exc()}")
            if str(e) == "Restarting after upgraded...":
                logging.info("Restarting after upgraded...")
            else:
                logging.info("Client tracing: {}".format(traceback.format_exc()))
        finally:
            print("finally")
            login_exit_file = os.path.join(ClientConstants.get_log_file_dir(), "exited.log")
            with open(login_exit_file, "w") as f:
                f.writelines(f"{os.getpid()}.")

            self.stop_agent()

            time.sleep(5)
            sys_utils.cleanup_all_fedml_client_login_processes(
                ClientConstants.CLIENT_LOGIN_PROGRAM, clean_process_group=False)
            sys.exit(1)

    def stop_agent(self):
        if self.run_process_event is not None:
            self.run_process_event.set()

        if self.model_device_server is not None:
            self.model_device_server.stop()
            self.model_device_server = None

        if self.model_device_client_list is not None:
            for model_client in self.model_device_client_list:
                model_client.stop()
            self.model_device_client_list.clear()
            self.model_device_client_list = None

        if self.mqtt_mgr is not None:
            try:
                for topic in self.subscribed_topics:
                    self.mqtt_mgr.unsubscribe_msg(topic)
            except Exception as e:
                logging.error(f"Unsubscribe topics error: {e}, Traceback: {traceback.format_exc()}")
                pass

            self.mqtt_mgr.loop_stop()
            self.mqtt_mgr.disconnect()

        self.release_message_center()

    def get_runner(self):
        runner = FedMLClientRunner(
            self.args, edge_id=self.edge_id, request_json=self.request_json,
            agent_config=self.agent_config, run_id=self.run_id,
            cuda_visible_gpu_ids_str=self.cuda_visible_gpu_ids_str
        )
        runner.edge_user_name = self.user_name
        runner.edge_extra_url = self.edge_extra_url
        runner.unique_device_id = self.unique_device_id
        runner.user_name = self.user_name
        runner.general_edge_id = self.general_edge_id
        runner.model_device_client_edge_id_list = self.model_device_client_edge_id_list
        runner.model_device_server_id = self.model_device_server_id
        return runner
