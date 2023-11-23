import base64
import copy
import json
import logging
import platform
import queue
import sys

import multiprocessing
from multiprocessing import Process, Queue
import os
import shutil
import stat
import subprocess
import threading

import time
import traceback
import urllib
import uuid
import zipfile
from os import listdir

import requests

import fedml
from ..scheduler_core.scheduler_matcher import SchedulerMatcher
from ..comm_utils.constants import SchedulerConstants
from ..comm_utils.job_utils import JobRunnerUtils
from ..comm_utils.run_process_utils import RunProcessUtils
from ....core.mlops.mlops_runtime_log import MLOpsRuntimeLog

from ....core.distributed.communication.mqtt.mqtt_manager import MqttManager
from ..comm_utils.yaml_utils import load_yaml_config
from ..slave.client_constants import ClientConstants
from ..master.server_constants import ServerConstants

from ....core.mlops.mlops_metrics import MLOpsMetrics

from ....core.mlops.mlops_configs import MLOpsConfigs
from ....core.mlops.mlops_runtime_log_daemon import MLOpsRuntimeLogDaemon
from ....core.mlops.mlops_status import MLOpsStatus
from ..comm_utils.sys_utils import get_sys_runner_info, get_python_program
from ..comm_utils import sys_utils
from .server_data_interface import FedMLServerDataInterface
from ....core.mlops.mlops_utils import MLOpsUtils
from ..scheduler_entry.constants import Constants
from ..model_scheduler.model_device_server import FedMLModelDeviceServerRunner
from ..model_scheduler.device_model_cards import FedMLModelCards
from ..model_scheduler import device_client_constants


class RunnerError(Exception):
    """ Runner stopped. """
    pass


class RunnerCompletedError(Exception):
    """ Runner completed. """
    pass


class FedMLServerRunner:
    FEDML_CLOUD_SERVER_PREFIX = "fedml-server-run-"

    def __init__(self, args, run_id=0, request_json=None, agent_config=None, edge_id=0):
        self.origin_fedml_config_object = None
        self.package_type = SchedulerConstants.JOB_PACKAGE_TYPE_DEFAULT
        self.local_api_process = None
        self.run_process_event = None
        self.run_process_event_map = dict()
        self.run_process_completed_event = None
        self.run_process_completed_event_map = dict()
        self.edge_status_queue = None
        self.edge_status_queue_map = dict()
        self.run_process = None
        self.run_process_map = dict()
        self.start_request_json = None
        self.server_docker_image = None
        self.cloud_server_name = None
        self.run_as_cloud_agent = False
        self.run_as_cloud_server = False
        self.run_as_edge_server_and_agent = False
        self.run_as_cloud_server_and_agent = False
        self.fedml_packages_base_dir = None
        self.fedml_packages_unzip_dir = None
        self.mqtt_mgr = None
        self.running_request_json = dict()
        self.run_id = run_id
        self.client_mqtt_mgr = None
        self.client_mqtt_is_connected = False
        self.client_mqtt_lock = None
        self.unique_device_id = None
        self.edge_id = edge_id
        self.server_agent_id = 0
        if request_json is not None:
            self.server_agent_id = request_json.get("server_id", 0)
        self.process = None
        self.args = args
        self.request_json = copy.deepcopy(request_json)
        self.version = args.version
        self.device_id = args.device_id
        self.cur_dir = os.path.split(os.path.realpath(__file__))[0]
        if args.current_running_dir is not None:
            self.cur_dir = args.current_running_dir

        image_version = self.version
        if image_version == "local":
            image_version = "dev"
        self.server_docker_base_image = "/fedml-device-image:" + image_version

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
        self.client_agent_active_list = dict()
        self.server_active_list = dict()
        self.run_status = None
        self.ntp_offset = MLOpsUtils.get_ntp_offset()
        self.runner_list = dict()
        self.enable_simulation_cloud_agent = False
        self.use_local_process_as_cloud_server = False

        self.model_device_server = None
        self.run_model_device_ids = dict()
        self.run_edge_ids = dict()
        self.run_edges_realtime_status = dict()

    def build_dynamic_constrain_variables(self, run_id, run_config):
        data_config = run_config.get("data_config", {})
        server_edge_id_list = self.request_json["edgeids"]
        is_using_local_data = 0
        private_data_dir = data_config.get("privateLocalData", "")
        synthetic_data_url = data_config.get("syntheticDataUrl", "")
        edges = self.request_json["edges"]
        # if private_data_dir is not None \
        #         and len(str(private_data_dir).strip(' ')) > 0:
        #     is_using_local_data = 1
        if private_data_dir is None or len(str(private_data_dir).strip(" ")) <= 0:
            params_config = run_config.get("parameters", None)
            private_data_dir = ServerConstants.get_data_dir()
        if synthetic_data_url is None or len(str(synthetic_data_url)) <= 0:
            synthetic_data_url = private_data_dir

        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.RUN_ID}"] = run_id
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.PRIVATE_LOCAL_DATA}"] = private_data_dir.replace(" ", "")
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.CLIENT_ID_LIST}"] = str(server_edge_id_list).replace(" ", "")
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.SYNTHETIC_DATA_URL}"] = synthetic_data_url.replace(" ", "")
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.IS_USING_LOCAL_DATA}"] = str(is_using_local_data)
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.CLIENT_NUM}"] = len(server_edge_id_list)
        client_objects = str(json.dumps(edges))
        client_objects = client_objects.replace(" ", "").replace("\n", "").replace('"', '\\"')
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.CLIENT_OBJECT_LIST}"] = client_objects
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.LOG_SERVER_URL}"] = self.agent_config["ml_ops_config"][
            "LOG_SERVER_URL"
        ]

    def unzip_file(self, zip_file, unzip_file_path) -> str:
        unziped_file_name = ""
        if zipfile.is_zipfile(zip_file):
            with zipfile.ZipFile(zip_file, "r") as zipf:
                zipf.extractall(unzip_file_path)
                unziped_file_name = zipf.namelist()[0]
        else:
            raise Exception("Invalid zip file {}".format(zip_file))

        return unziped_file_name

    def package_download_progress(self, count, blksize, filesize):
        self.check_runner_stop_event()

        downloaded = count * blksize
        downloaded = filesize if downloaded > filesize else downloaded
        progress = (downloaded / filesize * 100) if filesize != 0 else 0
        progress_int = int(progress)
        downloaded_kb = format(downloaded / 1024, '.2f')

        # since this hook funtion is stateless, we need a state to avoid printing progress repeatly
        if count == 0:
            self.prev_download_progress = 0
        if progress_int != self.prev_download_progress and progress_int % 5 == 0:
            self.prev_download_progress = progress_int
            logging.info("package downloaded size {} KB, progress {}%".format(downloaded_kb, progress_int))

    def retrieve_and_unzip_package(self, package_name, package_url):
        local_package_path = ServerConstants.get_package_download_dir()
        os.makedirs(local_package_path, exist_ok=True)
        filename, filename_without_extension, file_extension = ServerConstants.get_filename_and_extension(package_url)
        local_package_file = os.path.join(local_package_path, f"fedml_run_{self.run_id}_{filename_without_extension}")
        if os.path.exists(local_package_file):
            os.remove(local_package_file)
        urllib.request.urlretrieve(package_url, local_package_file, reporthook=self.package_download_progress)
        unzip_package_path = os.path.join(ClientConstants.get_package_unzip_dir(),
                                          f"unzip_fedml_run_{self.run_id}_{filename_without_extension}")
        try:
            shutil.rmtree(unzip_package_path, ignore_errors=True)
        except Exception as e:
            pass

        package_dir_name = self.unzip_file(local_package_file, unzip_package_path)  # Using unziped folder name
        unzip_package_full_path = os.path.join(unzip_package_path, package_dir_name)

        logging.info("local_package_file {}, unzip_package_path {}, unzip file full path {}".format(
            local_package_file, unzip_package_path, unzip_package_full_path))

        return unzip_package_full_path

    def update_local_fedml_config(self, run_id, run_config):
        packages_config = run_config["packages_config"]

        # Copy config file from the client
        server_package_name = packages_config.get("server", None)
        server_package_url = packages_config.get("serverUrl", None)
        unzip_package_path = self.retrieve_and_unzip_package(server_package_name, server_package_url)
        self.fedml_packages_unzip_dir = unzip_package_path
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
        log_file_dir = ServerConstants.get_log_file_dir()
        os.makedirs(log_file_dir, exist_ok=True)
        package_conf_object["dynamic_args"]["log_file_dir"] = log_file_dir

        # Save new config dictionary to local file
        fedml_updated_config_file = os.path.join(unzip_package_path, "conf", "fedml.yaml")
        ServerConstants.generate_yaml_doc(package_conf_object, fedml_updated_config_file)

        # Build dynamic arguments and set arguments to fedml config object
        if not self.build_dynamic_args(run_id, run_config, package_conf_object, unzip_package_path):
            return None, None

        return unzip_package_path, package_conf_object

    def build_dynamic_args(self, run_id, run_config, package_conf_object, base_dir):
        fedml_conf_file = package_conf_object["entry_config"]["conf_file"]
        fedml_conf_file_processed = str(fedml_conf_file).replace('\\', os.sep).replace('/', os.sep)
        fedml_conf_path = os.path.join(base_dir, "fedml", "config",
                                       os.path.basename(fedml_conf_file_processed))
        fedml_conf_object = load_yaml_config(fedml_conf_path)
        self.origin_fedml_config_object = fedml_conf_object.copy()
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
            fedml_conf_object["train_args"]["server_id"] = self.edge_id
            fedml_conf_object["train_args"]["server_agent_id"] = self.request_json.get("cloud_agent_id", self.edge_id)
            fedml_conf_object["train_args"]["group_server_id_list"] = self.request_json.get("group_server_id_list",
                                                                                            list())
        if fedml_conf_object.get("device_args", None) is not None:
            fedml_conf_object["device_args"]["worker_num"] = int(package_dynamic_args["client_num_in_total"])
        # fedml_conf_object["data_args"]["data_cache_dir"] = package_dynamic_args["data_cache_dir"]
        if fedml_conf_object.get("tracking_args", None) is not None:
            fedml_conf_object["tracking_args"]["log_file_dir"] = package_dynamic_args["log_file_dir"]
            fedml_conf_object["tracking_args"]["log_server_url"] = package_dynamic_args["log_server_url"]

        bootstrap_script_path = None
        env_args = fedml_conf_object.get("environment_args", None)
        if env_args is not None:
            bootstrap_script_file = env_args.get("bootstrap", None)
            if bootstrap_script_file is not None:
                bootstrap_script_file = str(bootstrap_script_file).replace('\\', os.sep).replace('/', os.sep)
                if platform.system() == 'Windows':
                    bootstrap_script_file = bootstrap_script_file.rstrip('.sh') + '.bat'
                if bootstrap_script_file is not None:
                    bootstrap_script_dir = os.path.join(base_dir, "fedml", os.path.dirname(bootstrap_script_file))
                    bootstrap_script_path = os.path.join(
                        bootstrap_script_dir, bootstrap_script_dir, os.path.basename(bootstrap_script_file)
                    )
        # try:
        #     os.makedirs(package_dynamic_args["data_cache_dir"], exist_ok=True)
        # except Exception as e:
        #     pass
        fedml_conf_object["dynamic_args"] = package_dynamic_args

        ServerConstants.generate_yaml_doc(fedml_conf_object, fedml_conf_path)

        is_bootstrap_run_ok = True
        try:
            if bootstrap_script_path is not None:
                if os.path.exists(bootstrap_script_path):
                    bootstrap_stat = os.stat(bootstrap_script_path)
                    if platform.system() == 'Windows':
                        os.chmod(bootstrap_script_path,
                                 bootstrap_stat.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                        bootstrap_scripts = "{}".format(bootstrap_script_path)
                    else:
                        os.chmod(bootstrap_script_path,
                                 bootstrap_stat.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                        bootstrap_scripts = "cd {}; ./{}".format(bootstrap_script_dir,
                                                                 os.path.basename(bootstrap_script_file))
                    bootstrap_scripts = str(bootstrap_scripts).replace('\\', os.sep).replace('/', os.sep)
                    logging.info("Bootstrap scripts are being executed...")
                    shell_cmd_list = list()
                    shell_cmd_list.append(bootstrap_scripts)
                    process, error_list = ServerConstants.execute_commands_with_live_logs(
                        shell_cmd_list, callback=self.callback_run_bootstrap)

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
            logging.error("Bootstrap scripts error: {}".format(traceback.format_exc()))

            is_bootstrap_run_ok = False

        return is_bootstrap_run_ok

    def callback_run_bootstrap(self, job_pid):
        ServerConstants.save_bootstrap_process(self.run_id, job_pid)

    def run(self, process_event, completed_event, edge_status_queue=None):
        print(f"Server runner process id {os.getpid()}, run id {self.run_id}")

        if platform.system() != "Windows":
            os.setsid()

        os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
        os.environ.setdefault('PYTHONWARNINGS', 'ignore:semaphore_tracker:UserWarning')

        self.run_process_event = process_event
        self.run_process_completed_event = completed_event
        try:
            MLOpsUtils.set_ntp_offset(self.ntp_offset)

            self.setup_client_mqtt_mgr()

            if self.run_as_cloud_server:
                topic_client_exit_train_with_exception = "flserver_agent/" + str(self.run_id) \
                                                         + "/client_exit_train_with_exception"
                self.client_mqtt_mgr.add_message_listener(topic_client_exit_train_with_exception,
                                                          self.callback_client_exit_train_with_exception)
                self.client_mqtt_mgr.subscribe_msg(topic_client_exit_train_with_exception)

            self.run_impl(edge_status_queue)
        except RunnerError:
            logging.info("Runner stopped.")
            self.mlops_metrics.report_server_training_status(self.run_id,
                                                             ServerConstants.MSG_MLOPS_SERVER_STATUS_KILLED)
        except RunnerCompletedError:
            logging.info("Runner completed.")
        except Exception as e:
            logging.error("Runner exits with exceptions. {}".format(traceback.format_exc()))
            self.mlops_metrics.report_server_training_status(self.run_id,
                                                             ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED)
        finally:
            logging.info("Release resources.")
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(self.run_id, self.edge_id)
            if self.mlops_metrics is not None:
                self.mlops_metrics.stop_sys_perf()
            time.sleep(3)
            ServerConstants.cleanup_run_process(self.run_id)
            ServerConstants.cleanup_learning_process(self.run_id)
            ServerConstants.cleanup_bootstrap_process(self.run_id)
            if not self.run_as_cloud_server:
                self.release_client_mqtt_mgr()
            if self.run_as_cloud_agent:
                self.stop_cloud_server()

    def check_runner_stop_event(self):
        if self.run_process_event.is_set():
            logging.info("Received stopping event.")
            raise RunnerError("Runner stopped")

        if self.run_process_completed_event.is_set():
            logging.info("Received completed event.")
            raise RunnerCompletedError("Runner completed")

    def deploy_model(self, serving_devices, request_json):
        run_config = request_json["run_config"]
        run_params = run_config.get("parameters", {})
        job_yaml = run_params.get("job_yaml", {})
        job_type = job_yaml.get("job_type", None)
        job_type = job_yaml.get("task_type", Constants.JOB_TASK_TYPE_TRAIN) if job_type is None else job_type
        if job_type == Constants.JOB_TASK_TYPE_DEPLOY or job_type == Constants.JOB_TASK_TYPE_SERVE:
            computing = job_yaml.get("computing", {})
            num_gpus = computing.get("minimum_num_gpus", 1)
            serving_args = run_params.get("serving_args", {})
            model_id = serving_args.get("model_id", None)
            model_name = serving_args.get("model_name", None)
            model_version = serving_args.get("model_version", None)
            model_storage_url = serving_args.get("model_storage_url", None)
            endpoint_name = serving_args.get("endpoint_name", None)
            endpoint_id = serving_args.get("endpoint_id", None)
            random = serving_args.get("random", "")
            random_out = sys_utils.random2(random, "FEDML@9999GREAT")
            random_list = random_out.split("FEDML@")
            device_type = device_client_constants.ClientConstants.login_role_list[
                device_client_constants.ClientConstants.LOGIN_MODE_FEDML_CLOUD_INDEX]
            FedMLModelCards.get_instance().deploy_model(
                model_name, device_type, json.dumps(serving_devices),
                "", random_list[1], None,
                in_model_id=model_id, in_model_version=model_version,
                endpoint_name=endpoint_name, endpoint_id=endpoint_id)

    def run_impl(self, edge_status_queue):
        run_id = self.request_json["runId"]
        run_config = self.request_json["run_config"]
        data_config = run_config["data_config"]
        edge_ids = self.request_json["edgeids"]

        self.check_runner_stop_event()

        self.run_id = run_id
        self.args.run_id = self.run_id
        MLOpsRuntimeLog.get_instance(self.args).init_logs(show_stdout_log=True)

        # report server running status
        self.mlops_metrics.report_server_training_status(run_id,
                                                         ServerConstants.MSG_MLOPS_SERVER_STATUS_STARTING,
                                                         running_json=self.start_request_json)

        logging.info("Detect all status of Edge ids: " + str(edge_ids))
        if not self.detect_edges_status(
                edge_status_queue, callback_when_edges_ready=self.send_training_request_to_edges):
            return

        if not self.should_continue_run_job(run_id):
            while True:
                self.check_runner_stop_event()
                self.reset_edges_status_map(run_id)
                if not self.detect_edges_status(edge_status_queue):
                    break
                time.sleep(10)
            return

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
            self.cleanup_run_when_starting_failed()
            self.send_exit_train_with_exception_request_to_edges(edge_ids, self.start_request_json)
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
            self.cleanup_run_when_starting_failed()
            self.send_exit_train_with_exception_request_to_edges(edge_ids, self.start_request_json)
            return
        os.chdir(os.path.join(unzip_package_path, "fedml"))

        self.check_runner_stop_event()

        logging.info("starting the aggregation process...")

        entry_file_full_path = os.path.join(unzip_package_path, "fedml", entry_file)
        conf_file_full_path = os.path.join(unzip_package_path, "fedml", conf_file)
        logging.info("                          ")
        logging.info("                          ")
        logging.info("====Your Run Logs Begin===")
        process, is_launch_task, error_list = self.execute_job_task(entry_file_full_path, conf_file_full_path, run_id)
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

                self.mlops_metrics.report_server_training_status(run_id,
                                                                 ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED)

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

            self.stop_run_when_starting_failed()

    def init_job_task(self, request_json):
        run_id = request_json["runId"]
        run_config = request_json["run_config"]
        edge_ids = request_json["edgeids"]
        run_params = run_config.get("parameters", {})
        job_yaml = run_params.get("job_yaml", None)
        server_id = request_json["server_id"]

        if job_yaml is not None:
            self.setup_listeners_for_edge_status(run_id, edge_ids, server_id)

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
                self.mlops_metrics.report_server_training_status(run_id,
                                                                 ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING,
                                                                 running_json=self.start_request_json)
                return False

        return True

    def execute_job_task(self, entry_file_full_path, conf_file_full_path, run_id):
        run_config = self.request_json["run_config"]
        run_params = run_config.get("parameters", {})
        job_yaml = run_params.get("job_yaml", {})
        job_yaml_default_none = run_params.get("job_yaml", None)
        job_api_key = job_yaml.get("run_api_key", None)
        job_api_key = job_yaml.get("fedml_run_dynamic_params", None) if job_api_key is None else job_api_key
        assigned_gpu_ids = run_params.get("gpu_ids", None)
        framework_type = job_yaml.get("framework_type", None)
        job_type = job_yaml.get("job_type", None)
        job_type = job_yaml.get("task_type", Constants.JOB_TASK_TYPE_TRAIN) if job_type is None else job_type
        conf_file_object = load_yaml_config(conf_file_full_path)
        entry_args_dict = conf_file_object.get("fedml_entry_args", {})
        entry_args = entry_args_dict.get("arg_items", None)

        executable_interpreter = ClientConstants.CLIENT_SHELL_PS \
            if platform.system() == ClientConstants.PLATFORM_WINDOWS else ClientConstants.CLIENT_SHELL_BASH

        if job_yaml_default_none is None:
            # Generate the job executing commands for previous federated learning (Compatibility)
            python_program = get_python_program()
            logging.info("Run the server: {} {} --cf {} --rank 0 --role server".format(
                python_program, entry_file_full_path, conf_file_full_path))
            entry_command = f"{python_program} {entry_file_full_path} --cf " \
                            f"{conf_file_full_path} --rank 0 --role server"
            shell_cmd_list = [entry_command]

            # Run the job executing commands for previous federated learning (Compatibility)
            process, error_list = ClientConstants.execute_commands_with_live_logs(
                shell_cmd_list, callback=self.callback_start_fl_job, should_write_log_file=False)
            is_launch_task = False
        else:
            self.check_runner_stop_event()

            self.mlops_metrics.report_server_training_status(run_id,
                                                             ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING,
                                                             running_json=self.start_request_json)

            # Generate the job executing commands
            job_executing_commands = JobRunnerUtils.generate_job_execute_commands(
                self.run_id, self.edge_id, self.version,
                self.package_type, executable_interpreter, entry_file_full_path,
                conf_file_object, entry_args, assigned_gpu_ids,
                job_api_key, 0, job_yaml=job_yaml_default_none)

            # Run the job executing commands
            logging.info(f"Run the server job with job id {self.run_id}, device id {self.edge_id}.")
            process, error_list = ServerConstants.execute_commands_with_live_logs(
                job_executing_commands, callback=self.start_job_perf, error_processor=self.job_error_processor)
            is_launch_task = True

        return process, is_launch_task, error_list

    def callback_start_fl_job(self, job_pid):
        ServerConstants.save_learning_process(self.run_id, job_pid)
        self.mlops_metrics.report_sys_perf(
            self.args, self.agent_config["mqtt_config"], job_process_id=job_pid)

    def start_job_perf(self, job_pid):
        ServerConstants.save_learning_process(self.run_id, job_pid)
        self.mlops_metrics.report_job_perf(self.args, self.agent_config["mqtt_config"], job_pid)

    def job_error_processor(self, error_list):
        error_str = "\n".join(error_list)
        raise Exception(f"Error occurs when running the job... {error_str}")

    def process_job_status(self, run_id):
        all_edges_is_finished = True
        any_edge_is_failed = False
        edge_id_status_dict = self.client_agent_active_list.get(f"{run_id}", {})
        server_id = edge_id_status_dict.get("server", 0)
        for edge_id, status in edge_id_status_dict.items():
            if edge_id == "server":
                continue

            if status is None:
                all_edges_is_finished = False
                continue

            if status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED:
                any_edge_is_failed = True
                break

            if status != ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED:
                all_edges_is_finished = False
                break

        if any_edge_is_failed:
            status = ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED
            self.report_server_status(run_id, server_id, status)
            return

        if all_edges_is_finished:
            status = ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED
            self.report_server_status(run_id, server_id, status)

    def report_server_status(self, run_id, server_id, status):
        self.mlops_metrics.report_server_id_status(run_id, status, edge_id=self.edge_id,
                                                   server_id=server_id, server_agent_id=self.edge_id)

    def reset_all_devices_status(self):
        edge_id_list = self.request_json["edgeids"]
        for edge_id in edge_id_list:
            self.mlops_metrics.common_broadcast_client_training_status(edge_id,
                                                                       ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE)

    def set_all_devices_status(self, status):
        edge_id_list = self.request_json["edgeids"]
        for edge_id in edge_id_list:
            self.mlops_metrics.common_broadcast_client_training_status(edge_id, status)

    def stop_run_entry(self):
        try:
            if self.run_process_event is not None:
                self.run_process_event.set()
            self.stop_run()
            # if self.run_process is not None:
            #     logging.info("Run will be stopped, waiting...")
            #     self.run_process.join()
        except Exception as e:
            pass

    def stop_run(self):
        edge_id_list = self.request_json["edgeids"]

        ServerConstants.cleanup_learning_process(self.run_id)
        ServerConstants.cleanup_bootstrap_process(self.run_id)
        ServerConstants.cleanup_run_process(self.run_id)

        self.send_training_stop_request_to_edges(edge_id_list, json.dumps(self.request_json))
        self.mlops_metrics.report_server_training_status(self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_KILLED)

        logging.info("stop run successfully.")

        try:
            local_package_path = ServerConstants.get_package_download_dir()
            for package_file in listdir(local_package_path):
                if os.path.basename(package_file).startswith("run_" + str(self.run_id)):
                    shutil.rmtree(os.path.join(local_package_path, package_file), ignore_errors=True)
        except Exception as e:
            pass

    def stop_run_when_starting_failed(self):
        edge_id_list = self.request_json["edgeids"]
        logging.error("edge ids {}".format(str(edge_id_list)))
        self.send_exit_train_with_exception_request_to_edges(edge_id_list, json.dumps(self.request_json))

        # logging.info("Stop run successfully when starting failed.")

        self.mlops_metrics.report_server_id_status(self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED)

        self.set_all_devices_status(ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED)

    def cleanup_run_when_finished(self, should_send_server_id_status=True):
        if self.run_as_cloud_agent:
            self.stop_cloud_server()

        #logging.info("Cleanup run successfully when finished.")

        self.mlops_metrics.broadcast_server_training_status(
            self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED, edge_id=self.edge_id
        )

        if should_send_server_id_status:
            self.mlops_metrics.report_server_id_status(self.run_id,
                                                       ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED,
                                                       edge_id=self.edge_id)

        try:
            self.mlops_metrics.stop_sys_perf()
        except Exception as ex:
            pass

        time.sleep(1)

        ServerConstants.cleanup_learning_process(self.run_id)
        ServerConstants.cleanup_bootstrap_process(self.run_id)

        try:
            local_package_path = ServerConstants.get_package_download_dir()
            for package_file in listdir(local_package_path):
                if os.path.basename(package_file).startswith("run_" + str(self.run_id)):
                    shutil.rmtree(os.path.join(local_package_path, package_file), ignore_errors=True)
        except Exception as e:
            pass

    def cleanup_run_when_starting_failed(self, should_send_server_id_status=True):
        if self.run_as_cloud_agent:
            self.stop_cloud_server()

        #logging.info("Cleanup run successfully when starting failed.")

        self.mlops_metrics.broadcast_server_training_status(self.run_id,
                                                            ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED,
                                                            edge_id=self.edge_id)

        if should_send_server_id_status:
            self.mlops_metrics.report_server_id_status(self.run_id,
                                                       ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED)

        try:
            self.mlops_metrics.stop_sys_perf()
        except Exception as ex:
            pass

        time.sleep(1)

        ServerConstants.cleanup_learning_process(self.run_id)
        ServerConstants.cleanup_bootstrap_process(self.run_id)

        try:
            local_package_path = ServerConstants.get_package_download_dir()
            for package_file in listdir(local_package_path):
                if os.path.basename(package_file).startswith("run_" + str(self.run_id)):
                    shutil.rmtree(os.path.join(local_package_path, package_file), ignore_errors=True)
        except Exception as e:
            pass

    def reset_edges_status_map(self, run_id):
        run_id_str = str(run_id)
        if self.run_edges_realtime_status.get(run_id_str, None) is not None:
            self.run_edges_realtime_status.pop(run_id_str)

    def should_process_async_cluster(self):
        run_config = self.request_json.get("run_config", {})
        run_params = run_config.get("parameters", {})
        common_args = run_params.get("common_args", {})
        enable_async_cluster = common_args.get("enable_async_cluster", False)
        async_check_timeout = common_args.get("async_check_timeout", 0)
        if enable_async_cluster:
            return True, async_check_timeout

        return False, async_check_timeout

    def detect_edges_status(self, edge_status_queue, callback_when_edges_ready=None):
        run_id = self.request_json["runId"]
        run_id_str = str(run_id)
        edge_id_list = self.request_json["edgeids"]

        # Init realtime status of all edges
        if self.run_edges_realtime_status.get(run_id_str, None) is None:
            self.run_edges_realtime_status[run_id_str] = dict()

        # Send status message to all edges
        for edge_id in edge_id_list:
            self.send_status_check_msg(run_id, edge_id, self.edge_id)
        time.sleep(3)

        total_sleep_seconds = 0
        status_check_sleep_seconds = 5
        allowed_status_check_sleep_seconds = 60 * 25
        allowed_status_check_sleep_seconds_for_async = 30
        while True:
            # Fetch edge info from the edge status queue, which will be added to realtime status map
            while True:
                self.check_runner_stop_event()

                try:
                    edge_info = edge_status_queue.get(block=False, timeout=3)
                    if edge_info is not None:
                        edge_id = edge_info.get("edge_id", None)
                        if edge_id is not None:
                            self.run_edges_realtime_status[run_id_str][edge_id] = edge_info
                except queue.Empty as e:  # If queue is empty, then break loop
                    break

            self.check_runner_stop_event()

            # Check all edges which don't send response status successfully
            # and retry to send the status checking message.
            active_edges_count = 0
            inactivate_edges = list()
            active_edge_info_dict = dict()
            for edge_id in edge_id_list:
                edge_info_dict = self.run_edges_realtime_status.get(run_id_str, {})
                edge_info = edge_info_dict.get(edge_id, None)
                if edge_info is not None:
                    active_edges_count += 1
                    active_edge_info_dict[str(edge_id)] = edge_info
                else:
                    inactivate_edges.append(edge_id)
                    self.send_status_check_msg(run_id, edge_id, self.edge_id)

            # If all edges are ready then send the starting job message to them
            if active_edges_count == len(edge_id_list):
                if callback_when_edges_ready is not None:
                    callback_when_edges_ready(active_edge_info_dict=active_edge_info_dict)
                break

            # Check if runner needs to stop and sleep specific time
            self.check_runner_stop_event()
            time.sleep(status_check_sleep_seconds)
            total_sleep_seconds += status_check_sleep_seconds

            # Check if the status response message has timed out to receive
            if total_sleep_seconds >= allowed_status_check_sleep_seconds:
                # If so, send failed message to MLOps and send exception message to all edges.
                logging.error(f"There are inactive edge devices. "
                              f"Inactivate edge id list is as follows. {inactivate_edges}")
                self.mlops_metrics.report_server_id_status(
                    run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED, edge_id=self.edge_id,
                    server_id=self.edge_id, server_agent_id=self.server_agent_id)
                self.send_exit_train_with_exception_request_to_edges(edge_id_list, json.dumps(self.request_json))
                return False

            # If we enable the mode for async cluster, then sleep some time and send messages to all clients.
            should_async, async_timeout = self.should_process_async_cluster()
            if should_async and total_sleep_seconds >= allowed_status_check_sleep_seconds_for_async:
                if async_timeout > allowed_status_check_sleep_seconds_for_async:
                    time.sleep(async_timeout-allowed_status_check_sleep_seconds_for_async)
                self.send_training_request_to_edges()
                return True

        return True

    def send_status_check_msg(self, run_id, edge_id, server_id):
        topic_get_model_device_id = "server/client/request_device_info/" + str(edge_id)
        payload = {"server_id": server_id, "run_id": run_id}
        self.client_mqtt_mgr.send_message(topic_get_model_device_id, json.dumps(payload))

    def send_training_request_to_edges(self, active_edge_info_dict=None):
        run_id = self.request_json["runId"]
        edge_id_list = self.request_json["edgeids"]
        run_config = self.request_json.get("run_config", {})
        run_params = run_config.get("parameters", {})
        job_yaml = run_params.get("job_yaml", {})
        job_yaml_default_none = run_params.get("job_yaml", None)
        computing = job_yaml.get("computing", {})
        request_num_gpus = computing.get("minimum_num_gpus", None)

        logging.info("Send training request to Edge ids: " + str(edge_id_list))

        if job_yaml_default_none is not None and request_num_gpus is not None and \
                int(request_num_gpus) > 0 and active_edge_info_dict is not None:
            SchedulerMatcher.parse_and_print_gpu_info_for_all_edges(active_edge_info_dict, show_gpu_list=True)

            # Match and assign gpus to each device
            assigned_gpu_num_dict, assigned_gpu_ids_dict = SchedulerMatcher.match_and_assign_gpu_resources_to_devices(
                request_num_gpus, edge_id_list, active_edge_info_dict)
            if assigned_gpu_num_dict is None or assigned_gpu_ids_dict is None:
                # If no resources available, send failed message to MLOps and send exception message to all edges.
                gpu_count, gpu_available_count = SchedulerMatcher.parse_and_print_gpu_info_for_all_edges(
                    active_edge_info_dict, should_print=True)
                logging.error(f"No resources available."
                              f"Total available GPU count {gpu_available_count} is less than "
                              f"request GPU count {request_num_gpus}")
                self.mlops_metrics.report_server_id_status(
                    run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED, edge_id=self.edge_id,
                    server_id=self.edge_id, server_agent_id=self.server_agent_id)
                self.send_exit_train_with_exception_request_to_edges(edge_id_list, json.dumps(self.request_json))
                return

            # Generate master node addr and port
            master_node_addr, master_node_port = SchedulerMatcher.get_master_node_info(edge_id_list, active_edge_info_dict)

            # Generate new edge id list after matched
            edge_id_list = SchedulerMatcher.generate_new_edge_list_for_gpu_matching(assigned_gpu_num_dict)
            if len(edge_id_list) <= 0:
                gpu_count, gpu_available_count = SchedulerMatcher.parse_and_print_gpu_info_for_all_edges(
                    active_edge_info_dict, should_print=True)
                logging.error(f"Request parameter for GPU num is invalid."
                              f"Total available GPU count {gpu_available_count}."
                              f"Request GPU num {request_num_gpus}")
                self.mlops_metrics.report_server_id_status(
                    run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED, edge_id=self.edge_id,
                    server_id=self.edge_id, server_agent_id=self.server_agent_id)
                self.send_exit_train_with_exception_request_to_edges(edge_id_list, json.dumps(self.request_json))
                return

        client_rank = 1
        for edge_id in edge_id_list:
            topic_start_train = "flserver_agent/" + str(edge_id) + "/start_train"
            logging.info("start_train: send topic " + topic_start_train + " to client...")
            request_json = self.request_json
            request_json["client_rank"] = client_rank
            client_rank += 1

            if active_edge_info_dict is not None:
                edge_info = active_edge_info_dict.get(str(edge_id), {})
                model_master_device_id = edge_info.get("master_device_id", None)
                model_slave_device_id = edge_info.get("slave_device_id", None)

                if job_yaml_default_none is not None and request_num_gpus is not None:
                    request_json["scheduler_match_info"] = SchedulerMatcher.generate_match_info_for_scheduler(
                        edge_id, edge_id_list, master_node_addr, master_node_port,
                        assigned_gpu_num_dict, assigned_gpu_ids_dict,
                        model_master_device_id=model_master_device_id,
                        model_slave_device_id=model_slave_device_id
                    )

            self.client_mqtt_mgr.send_message(topic_start_train, json.dumps(request_json))

    def setup_listeners_for_edge_status(self, run_id, edge_ids, server_id):
        self.client_agent_active_list[f"{run_id}"] = dict()
        self.client_agent_active_list[f"{run_id}"][f"server"] = server_id
        for edge_id in edge_ids:
            self.client_agent_active_list[f"{run_id}"][f"{edge_id}"] = ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE
            edge_status_topic = "fl_client/flclient_agent_" + str(edge_id) + "/status"
            self.mqtt_mgr.add_message_listener(edge_status_topic, self.callback_edge_status)
            self.mqtt_mgr.subscribe_msg(edge_status_topic)

    def remove_listeners_for_edge_status(self, edge_ids=None):
        if edge_ids is None:
            edge_ids = self.request_json["edgeids"]

        for edge_id in edge_ids:
            edge_status_topic = "fl_client/flclient_agent_" + str(edge_id) + "/status"
            self.mqtt_mgr.remove_message_listener(edge_status_topic)

    def callback_edge_status(self, topic, payload):
        payload_json = json.loads(payload)
        run_id = payload_json.get("run_id", None)
        edge_id = payload_json.get("edge_id", None)
        status = payload_json.get("status", None)
        if run_id is not None and edge_id is not None:
            active_item_dict = self.client_agent_active_list.get(f"{run_id}", None)
            if active_item_dict is None:
                return
            self.client_agent_active_list[f"{run_id}"][f"{edge_id}"] = status
            self.process_job_status(run_id)

    def ota_upgrade(self, payload, request_json):
        run_id = request_json["runId"]
        force_ota = False
        ota_version = None

        try:
            run_config = request_json.get("run_config", None)
            parameters = run_config.get("parameters", None)
            common_args = parameters.get("common_args", None)
            force_ota = common_args.get("force_ota", False)
            ota_version = common_args.get("ota_version", None)
        except Exception as e:
            pass

        if force_ota and ota_version is not None:
            should_upgrade = True if ota_version != fedml.__version__ else False
            upgrade_version = ota_version
        else:
            try:
                fedml_is_latest_version, local_ver, remote_ver = sys_utils.check_fedml_is_latest_version(self.version)
            except Exception as e:
                return

            should_upgrade = False if fedml_is_latest_version else True
            upgrade_version = remote_ver

        if should_upgrade:
            job_obj = FedMLServerDataInterface.get_instance().get_job_by_id(run_id)
            if job_obj is None:
                FedMLServerDataInterface.get_instance(). \
                    save_started_job(run_id, self.edge_id, time.time(),
                                     ServerConstants.MSG_MLOPS_SERVER_STATUS_UPGRADING,
                                     ServerConstants.MSG_MLOPS_SERVER_STATUS_UPGRADING,
                                     payload)
                self.mlops_metrics.report_server_training_status(run_id,
                                                                 ServerConstants.MSG_MLOPS_SERVER_STATUS_UPGRADING)
            logging.info(f"Upgrade to version {upgrade_version} ...")

            sys_utils.do_upgrade(self.version, upgrade_version)

            raise Exception("Restarting after upgraded...")

    def callback_start_train(self, topic=None, payload=None):
        print("callback_start_train: ")
        try:
            _, _ = MLOpsConfigs.get_instance(self.args).fetch_configs()
        except Exception as e:
            pass

        # get training params
        if self.run_as_cloud_server:
            message_bytes = payload.encode("ascii")
            base64_bytes = base64.b64decode(message_bytes)
            payload = base64_bytes.decode("ascii")

        request_json = json.loads(payload)
        is_retain = request_json.get("is_retain", False)
        if is_retain:
            return

        # Process the log
        run_id = request_json["runId"]
        run_id_str = str(run_id)
        if self.run_as_edge_server_and_agent or self.enable_simulation_cloud_agent:
            # Start log processor for current run
            self.args.run_id = run_id
            self.args.edge_id = self.edge_id
            MLOpsRuntimeLog.get_instance(self.args).init_logs(show_stdout_log=True)
            MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(
                run_id, self.edge_id, SchedulerConstants.get_log_source(request_json))
            logging.info("start the log processor.")
        elif self.run_as_cloud_agent:
            # Start log processor for current run
            MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(
                run_id, request_json.get("server_id", "0"), SchedulerConstants.get_log_source(request_json)
            )
        elif self.run_as_cloud_server:
            self.server_agent_id = request_json.get("cloud_agent_id", self.edge_id)
            run_id = request_json["runId"]
            run_id_str = str(run_id)

            # Start log processor for current run
            self.args.run_id = run_id
            MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(
                run_id, self.edge_id, SchedulerConstants.get_log_source(request_json))

        logging.info("callback_start_train payload: {}".format(payload))
        logging.info(
            f"FedMLDebug - Receive: topic ({topic}), payload ({payload})"
        )

        if not self.run_as_cloud_agent and not self.run_as_cloud_server:
            self.ota_upgrade(payload, request_json)

        self.start_request_json = payload
        self.run_id = run_id
        ServerConstants.save_runner_infos(self.args.device_id + "." + self.args.os_name, self.edge_id, run_id=run_id)

        # Start server with multiprocessing mode
        self.request_json = request_json
        self.running_request_json[run_id_str] = request_json
        edge_id_list = request_json.get("edgeids", list())
        self.run_edge_ids[run_id_str] = edge_id_list

        logging.info("subscribe the client exception message.")

        # Setup MQTT message listener for run exception
        if self.run_as_edge_server_and_agent or self.enable_simulation_cloud_agent:
            topic_client_exit_train_with_exception = "flserver_agent/" + run_id_str + \
                                                     "/client_exit_train_with_exception"
            self.mqtt_mgr.add_message_listener(topic_client_exit_train_with_exception,
                                               self.callback_client_exit_train_with_exception)
            self.mqtt_mgr.subscribe_msg(topic_client_exit_train_with_exception)

        if self.run_as_edge_server_and_agent or self.enable_simulation_cloud_agent:
            self.init_job_task(request_json)

            self.args.run_id = run_id

            server_runner = FedMLServerRunner(
                self.args, run_id=run_id, request_json=request_json, agent_config=self.agent_config
            )
            server_runner.run_as_edge_server_and_agent = self.run_as_edge_server_and_agent
            server_runner.edge_id = self.edge_id
            server_runner.start_request_json = self.start_request_json
            self.run_process_event_map[run_id_str] = multiprocessing.Event()
            self.run_process_event_map[run_id_str].clear()
            server_runner.run_process_event = self.run_process_event_map[run_id_str]
            self.run_process_completed_event_map[run_id_str] = multiprocessing.Event()
            self.run_process_completed_event_map[run_id_str].clear()
            server_runner.run_process_completed_event = self.run_process_completed_event_map[run_id_str]
            if self.edge_status_queue_map.get(run_id_str, None) is None:
                self.edge_status_queue_map[run_id_str] = Queue()
            server_runner.edge_status_queue = self.edge_status_queue_map[run_id_str]
            logging.info("start the runner process.")
            self.run_process_map[run_id_str] = Process(target=server_runner.run, args=(
                self.run_process_event_map[run_id_str], self.run_process_completed_event_map[run_id_str],
                self.edge_status_queue_map[run_id_str], ))
            self.run_process_map[run_id_str].start()
            ServerConstants.save_run_process(run_id, self.run_process_map[run_id_str].pid)
        elif self.run_as_cloud_agent:
            self.init_job_task(request_json)

            server_runner = FedMLServerRunner(
                self.args, run_id=run_id, request_json=request_json, agent_config=self.agent_config
            )
            server_runner.run_as_cloud_agent = self.run_as_cloud_agent
            server_runner.start_request_json = self.start_request_json
            self.run_process_event_map[run_id_str] = multiprocessing.Event()
            self.run_process_event_map[run_id_str].clear()
            server_runner.run_process_event = self.run_process_event_map[run_id_str]

            if not self.use_local_process_as_cloud_server:
                self.run_process_map[run_id_str] = Process(target=server_runner.start_cloud_server_process_entry)
                self.run_process_map[run_id_str].start()
            else:
                message_bytes = json.dumps(self.request_json).encode("ascii")
                base64_bytes = base64.b64encode(message_bytes)
                runner_cmd_encoded = base64_bytes.decode("ascii")
                logging.info("runner_cmd_encoded: {}".format(runner_cmd_encoded))

                cloud_device_id = request_json.get("cloudServerDeviceId", "0")

                pip_source_dir = os.path.dirname(__file__)
                login_cmd = os.path.join(pip_source_dir, "server_login.py")
                self.run_process_map[run_id_str] = ServerConstants.exec_console_with_shell_script_list(
                    [
                        get_python_program(),
                        "-W",
                        "ignore",
                        login_cmd,
                        "-t",
                        "login",
                        "-r",
                        "cloud_server",
                        "-u",
                        str(self.args.user),
                        "-v",
                        self.version,
                        "-id",
                        cloud_device_id,
                        "-rc",
                        runner_cmd_encoded
                    ])
                time.sleep(1)

            ServerConstants.save_run_process(run_id, self.run_process_map[run_id_str].pid)
        elif self.run_as_cloud_server:
            self.server_agent_id = request_json.get("cloud_agent_id", self.edge_id)
            self.start_request_json = json.dumps(request_json)
            run_id = request_json["runId"]
            run_id_str = str(run_id)

            self.init_job_task(request_json)

            self.args.run_id = run_id

            server_runner = FedMLServerRunner(
                self.args, run_id=run_id, request_json=request_json, agent_config=self.agent_config
            )
            server_runner.run_as_edge_server_and_agent = self.run_as_edge_server_and_agent
            server_runner.edge_id = self.edge_id
            server_runner.server_agent_id = self.server_agent_id
            server_runner.start_request_json = self.start_request_json
            self.run_process_event_map[run_id_str] = multiprocessing.Event()
            self.run_process_event_map[run_id_str].clear()
            server_runner.run_process_event = self.run_process_event_map[run_id_str]
            self.run_process_completed_event_map[run_id_str] = multiprocessing.Event()
            self.run_process_completed_event_map[run_id_str].clear()
            server_runner.run_process_completed_event = self.run_process_completed_event_map[run_id_str]
            if self.edge_status_queue_map.get(run_id_str, None) is None:
                self.edge_status_queue_map[run_id_str] = Queue()
            server_runner.edge_status_queue = self.edge_status_queue_map[run_id_str]
            logging.info("start the runner process.")
            server_process = Process(target=server_runner.run, args=(
                self.run_process_event_map[run_id_str], self.run_process_completed_event_map[run_id_str],
                self.edge_status_queue_map[run_id_str], ))
            server_process.start()
            # ServerConstants.save_run_process(run_id, self.run_process_map[run_id_str].pid)

    def start_cloud_server_process_entry(self):
        try:
            self.start_cloud_server_process()
        except Exception as e:
            pass

    def start_cloud_server_process(self):
        run_config = self.request_json["run_config"]
        packages_config = run_config["packages_config"]
        self.start_cloud_server(packages_config)

    def start_cloud_server(self, packages_config):
        server_id = self.request_json["server_id"]
        self.cloud_server_name = FedMLServerRunner.FEDML_CLOUD_SERVER_PREFIX + str(self.run_id) + "-" + str(server_id)
        self.server_docker_image = (
                self.agent_config["docker_config"]["registry_server"]
                + self.agent_config["docker_config"]["registry_dir"]
                + self.server_docker_base_image
        )

        logging.info("docker image {}".format(self.server_docker_image))
        # logging.info("file_sys_driver {}".format(self.agent_config["docker_config"]["file_sys_driver"]))

        registry_secret_cmd = (
                "kubectl create namespace fedml-devops-aggregator-"
                + self.version
                + ";kubectl -n fedml-devops-aggregator-"
                + self.version
                + " delete secret secret-"
                + self.cloud_server_name
                + " ;kubectl create secret docker-registry secret-"
                + self.cloud_server_name
                + " --docker-server="
                + self.agent_config["docker_config"]["registry_server"]
                + " --docker-username="
                + self.agent_config["docker_config"]["user_name"]
                + " --docker-password=$(aws ecr-public get-login-password --region "
                + self.agent_config["docker_config"]["public_cloud_region"]
                + ")"
                + " --docker-email=fedml@fedml.ai -n fedml-devops-aggregator-"
                + self.version
        )
        logging.info("Create secret cmd: " + registry_secret_cmd)
        os.system(registry_secret_cmd)

        message_bytes = json.dumps(self.request_json).encode("ascii")
        base64_bytes = base64.b64encode(message_bytes)
        runner_cmd_encoded = base64_bytes.decode("ascii")
        logging.info("runner_cmd_encoded: {}".format(runner_cmd_encoded))
        # logging.info("runner_cmd_decoded: {}".format(base64.b64decode(runner_cmd_encoded).decode()))
        cur_dir = os.path.dirname(__file__)
        run_deployment_cmd = (
                "export FEDML_AGGREGATOR_NAME="
                + self.cloud_server_name
                + ";export FEDML_AGGREGATOR_SVC="
                + self.cloud_server_name
                + ";export FEDML_AGGREGATOR_VERSION="
                + self.version
                + ';export FEDML_AGGREGATOR_IMAGE_PATH="'
                + self.server_docker_image
                + '"'
                + ";export FEDML_CONF_ID="
                + self.cloud_server_name
                + ";export FEDML_DATA_PV_ID="
                + self.cloud_server_name
                + ";export FEDML_DATA_PVC_ID="
                + self.cloud_server_name
                + ";export FEDML_REGISTRY_SECRET_SUFFIX="
                + self.cloud_server_name
                + ";export FEDML_ACCOUNT_ID=0"
                + ";export FEDML_SERVER_DEVICE_ID="
                + self.request_json.get("cloudServerDeviceId", "0")
                + ";export FEDML_VERSION="
                + self.version
                + ";export FEDML_PACKAGE_NAME="
                + packages_config.get("server", "")
                + ";export FEDML_PACKAGE_URL="
                + packages_config.get("serverUrl", "")
                + ";export FEDML_RUNNER_CMD="
                + runner_cmd_encoded
                + ";envsubst < "
                + os.path.join(cur_dir, "templates", "fedml-server-deployment.yaml")
                + " | kubectl apply -f - "
        )
        logging.info("FedMLServerRunner.run with k8s: " + run_deployment_cmd)
        os.system(run_deployment_cmd)

    def stop_cloud_server_process_entry(self):
        try:
            self.setup_client_mqtt_mgr()
            self.stop_cloud_server_process()
        except Exception as e:
            pass
        finally:
            self.release_client_mqtt_mgr()

    def stop_cloud_server_process(self):
        self.stop_cloud_server()

        self.stop_run()

    def stop_cloud_server(self):
        self.cloud_server_name = FedMLServerRunner.FEDML_CLOUD_SERVER_PREFIX + str(self.run_id) \
                                 + "-" + str(self.edge_id)
        self.server_docker_image = (
                self.agent_config["docker_config"]["registry_server"]
                + self.agent_config["docker_config"]["registry_dir"]
                + self.server_docker_base_image
        )
        delete_deployment_cmd = (
                "export FEDML_AGGREGATOR_NAME="
                + self.cloud_server_name
                + ";export FEDML_AGGREGATOR_SVC="
                + self.cloud_server_name
                + ";export FEDML_AGGREGATOR_VERSION="
                + self.version
                + ';export FEDML_AGGREGATOR_IMAGE_PATH="'
                + self.server_docker_image
                + '"'
                + ";export FEDML_CONF_ID="
                + self.cloud_server_name
                + ";export FEDML_DATA_PV_ID="
                + self.cloud_server_name
                + ";export FEDML_DATA_PVC_ID="
                + self.cloud_server_name
                + ";export FEDML_REGISTRY_SECRET_SUFFIX="
                + self.cloud_server_name
                + ";kubectl -n fedml-devops-aggregator-"
                + self.version
                + " delete deployment "
                + self.cloud_server_name
                + ";kubectl -n fedml-devops-aggregator-"
                + self.version
                + " delete svc "
                + self.cloud_server_name
                + ";kubectl -n fedml-devops-aggregator-"
                + self.version
                + " delete secret secret-"
                + self.cloud_server_name
        )
        logging.info("FedMLServerRunner.stop_run with k8s: " + delete_deployment_cmd)
        os.system(delete_deployment_cmd)

    def on_client_mqtt_disconnected(self, mqtt_client_object):
        if self.client_mqtt_lock is None:
            self.client_mqtt_lock = threading.Lock()

        self.client_mqtt_lock.acquire()
        self.client_mqtt_is_connected = False
        self.client_mqtt_lock.release()

        # logging.info("on_client_mqtt_disconnected: {}.".format(self.client_mqtt_is_connected))

    def on_client_mqtt_connected(self, mqtt_client_object):
        if self.mlops_metrics is None:
            self.mlops_metrics = MLOpsMetrics()

        self.mlops_metrics.set_messenger(self.client_mqtt_mgr)
        self.mlops_metrics.run_id = self.run_id
        self.mlops_metrics.edge_id = self.edge_id
        self.mlops_metrics.server_agent_id = self.server_agent_id

        if self.client_mqtt_lock is None:
            self.client_mqtt_lock = threading.Lock()

        self.client_mqtt_lock.acquire()
        self.client_mqtt_is_connected = True
        self.client_mqtt_lock.release()

        # logging.info("on_client_mqtt_connected: {}.".format(self.client_mqtt_is_connected))

    def setup_client_mqtt_mgr(self):
        if self.client_mqtt_mgr is not None:
            return

        if self.client_mqtt_lock is None:
            self.client_mqtt_lock = threading.Lock()

        # logging.info(
        #     "server agent config: {},{}".format(
        #         self.agent_config["mqtt_config"]["BROKER_HOST"], self.agent_config["mqtt_config"]["BROKER_PORT"]
        #     )
        # )

        self.client_mqtt_mgr = MqttManager(
            self.agent_config["mqtt_config"]["BROKER_HOST"],
            self.agent_config["mqtt_config"]["BROKER_PORT"],
            self.agent_config["mqtt_config"]["MQTT_USER"],
            self.agent_config["mqtt_config"]["MQTT_PWD"],
            self.agent_config["mqtt_config"]["MQTT_KEEPALIVE"],
            "FedML_ServerAgent_Metrics_{}_{}_{}".format(self.args.current_device_id,
                                                        str(os.getpid()),
                                                        str(uuid.uuid4()))
        )
        self.client_mqtt_mgr.add_connected_listener(self.on_client_mqtt_connected)
        self.client_mqtt_mgr.add_disconnected_listener(self.on_client_mqtt_disconnected)
        self.client_mqtt_mgr.connect()
        self.client_mqtt_mgr.loop_start()

        if self.mlops_metrics is None:
            self.mlops_metrics = MLOpsMetrics()
        self.mlops_metrics.set_messenger(self.client_mqtt_mgr)
        self.mlops_metrics.run_id = self.run_id
        self.mlops_metrics.edge_id = self.edge_id
        self.mlops_metrics.server_agent_id = self.server_agent_id

    def release_client_mqtt_mgr(self):
        try:
            if self.client_mqtt_mgr is not None:
                self.client_mqtt_mgr.loop_stop()
                self.client_mqtt_mgr.disconnect()

            self.client_mqtt_lock.acquire()
            if self.client_mqtt_mgr is not None:
                self.client_mqtt_is_connected = False
                self.client_mqtt_mgr = None
            self.client_mqtt_lock.release()
        except Exception:
            pass

    def send_training_stop_request_to_edges(self, edge_id_list, payload):
        for edge_id in edge_id_list:
            topic_stop_train = "flserver_agent/" + str(edge_id) + "/stop_train"
            logging.info("stop_train: send topic " + topic_stop_train)
            self.client_mqtt_mgr.send_message(topic_stop_train, payload)
            self.mlops_metrics.common_broadcast_client_training_status(edge_id,
                                                                       ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED)

    def send_exit_train_with_exception_request_to_edges(self, edge_id_list, payload):
        for edge_id in edge_id_list:
            topic_exit_train = "flserver_agent/" + str(edge_id) + "/exit_train_with_exception"
            logging.error("exit_train_with_exception: send topic " + topic_exit_train)
            payload_obj = json.loads(payload)
            payload_obj["server_id"] = self.edge_id
            self.client_mqtt_mgr.send_message(topic_exit_train, json.dumps(payload_obj))
            self.mlops_metrics.common_broadcast_client_training_status(edge_id,
                                                                       ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED)

    def callback_stop_train(self, topic, payload):
        # logging.info("callback_stop_train: topic = %s, payload = %s" % (topic, payload))
        logging.info(
            f"FedMLDebug - Receive: topic ({topic}), payload ({payload})"
        )

        request_json = json.loads(payload)
        is_retain = request_json.get("is_retain", False)
        if is_retain:
            return
        run_id = request_json.get("runId", None)
        if run_id is None:
            run_id = request_json.get("id", None)

        edge_id_list = request_json["edgeids"]
        server_id = request_json.get("serverId", None)
        if server_id is None:
            server_id = request_json.get("server_id", None)

        if run_id is None or server_id is None:
            logging("Json format is not correct!")
            return

        # logging.info("Stop run with multiprocessing.")

        self.remove_listeners_for_edge_status()

        # Stop server with multiprocessing mode
        run_id_str = str(run_id)
        stop_request_json = self.running_request_json.get(run_id_str, None)
        if stop_request_json is None:
            stop_request_json = request_json
        if self.run_as_edge_server_and_agent or self.enable_simulation_cloud_agent:
            server_runner = FedMLServerRunner(
                self.args, run_id=run_id, request_json=stop_request_json, agent_config=self.agent_config,
                edge_id=self.edge_id
            )
            server_runner.run_as_edge_server_and_agent = self.run_as_edge_server_and_agent
            server_runner.run_process_event = self.run_process_event_map.get(run_id_str, None)
            server_runner.run_process = self.run_process_map.get(run_id_str, None)
            server_runner.client_mqtt_mgr = self.client_mqtt_mgr
            server_runner.mlops_metrics = self.mlops_metrics
            server_runner.stop_run_entry()

            # Stop log processor for current run
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, self.edge_id)
        elif self.run_as_cloud_agent:
            if not self.use_local_process_as_cloud_server:
                server_runner = FedMLServerRunner(
                    self.args, run_id=run_id, request_json=stop_request_json, agent_config=self.agent_config,
                    edge_id=server_id
                )
                server_runner.run_as_cloud_agent = self.run_as_cloud_agent
                server_runner.run_process_event = self.run_process_event_map.get(run_id_str, None)
                server_runner.run_process = self.run_process_map.get(run_id_str, None)
                Process(target=server_runner.stop_cloud_server_process_entry).start()
            else:
                ServerConstants.cleanup_run_process(run_id)

            # Stop log processor for current run
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, server_id)
        elif self.run_as_cloud_server:
            # Stop log processor for current run
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, self.edge_id)
            pass

        if self.running_request_json.get(run_id_str, None) is not None:
            self.running_request_json.pop(run_id_str)

        if self.run_process_map.get(run_id_str, None) is not None:
            self.run_process_map.pop(run_id_str)

    def exit_run_with_exception_entry(self):
        try:
            self.setup_client_mqtt_mgr()
            self.exit_run_with_exception()
        except Exception as e:
            self.release_client_mqtt_mgr()

        finally:
            self.release_client_mqtt_mgr()

    def exit_run_with_exception(self):
        logging.error("Exit run successfully.")

        ServerConstants.cleanup_learning_process(self.run_id)
        ServerConstants.cleanup_run_process(self.run_id)
        ServerConstants.cleanup_bootstrap_process(self.run_id)

        self.mlops_metrics.report_server_id_status(self.run_id,
                                                   ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED)

        time.sleep(1)

    def callback_exit_train_with_exception(self, topic, payload):
        # logging.info("callback_exit_train_with_exception: topic = %s, payload = %s" % (topic, payload))
        logging.info(
            f"FedMLDebug - Receive: topic ({topic}), payload ({payload})"
        )

        request_json = json.loads(payload)
        is_retain = request_json.get("is_retain", False)
        if is_retain:
            return
        run_id = request_json.get("runId", None)
        if run_id is None:
            run_id = request_json.get("run_id", None)
            if run_id is None:
                run_id = request_json.get("id", None)

        if run_id is None:
            return

        edge_ids = request_json.get("edgeids", None)

        self.send_exit_train_with_exception_request_to_edges(edge_ids, payload)

        # Stop server with multiprocessing mode
        self.request_json = request_json
        server_runner = FedMLServerRunner(
            self.args, edge_id=self.edge_id, request_json=request_json, agent_config=self.agent_config, run_id=run_id
        )
        try:
            Process(target=server_runner.exit_run_with_exception_entry).start()
        except Exception as e:
            pass

    def callback_client_exit_train_with_exception(self, topic, payload):
        # logging.info("callback_client_exit_train_with_exception: topic = %s, payload = %s" % (topic, payload))

        request_json = json.loads(payload)
        run_id = request_json.get("run_id", None)
        edge_id = request_json.get("edge_id", None)
        msg = request_json.get("msg", None)
        if run_id is None:
            logging.info("callback_client_exit_train_with_exception run id is none")
            return

        job = FedMLServerDataInterface.get_instance().get_job_by_id(run_id)
        if job is not None and job.running_json is not None and job.running_json != "":
            job_json_obj = json.loads(job.running_json)
            edge_ids = job_json_obj.get("edgeids", None)

            if msg is not None:
                logging.error(f"{msg}")

            self.mlops_metrics.broadcast_server_training_status(run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED)

            self.send_exit_train_with_exception_request_to_edges(edge_ids, job.running_json)

            self.exit_run_with_exception()

    def callback_runner_id_status(self, topic, payload):
        # logging.info("callback_runner_id_status: topic = %s, payload = %s" % (topic, payload))
        # logging.info(
        #     f"FedMLDebug - Receive: topic ({topic}), payload ({payload})"
        # )

        request_json = json.loads(payload)
        is_retain = request_json.get("is_retain", False)
        if is_retain:
            return
        run_id = request_json["run_id"]
        status = request_json["status"]
        edge_id = request_json["edge_id"]
        server_id = request_json.get("server_id", None)
        run_id_str = str(run_id)

        if (
                status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED
                or status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED
        ):
            completed_event = self.run_process_completed_event_map.get(run_id_str, None)
            if completed_event is not None:
                completed_event.set()

            # Stop server with multiprocessing mode
            stop_request_json = self.running_request_json.get(run_id_str, None)
            if stop_request_json is None:
                stop_request_json = request_json
            if self.run_as_edge_server_and_agent or self.enable_simulation_cloud_agent:
                server_runner = FedMLServerRunner(
                    self.args, run_id=run_id, request_json=stop_request_json, agent_config=self.agent_config
                )
                server_runner.edge_id = self.edge_id
                server_runner.run_as_edge_server_and_agent = self.run_as_edge_server_and_agent
                server_runner.run_status = status
                server_runner.client_mqtt_mgr = self.client_mqtt_mgr
                server_runner.mlops_metrics = self.mlops_metrics
                server_runner.cleanup_client_with_status()

                run_process = self.run_process_map.get(run_id_str, None)
                if run_process is not None:
                    if run_process.pid is not None:
                        RunProcessUtils.kill_process(run_process.pid)

                    self.run_process_map.pop(run_id_str)

                # Stop log processor for current run
                MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, self.edge_id)
            elif self.run_as_cloud_agent:
                if not self.use_local_process_as_cloud_server:
                    server_runner = FedMLServerRunner(
                        self.args, run_id=run_id, request_json=stop_request_json, agent_config=self.agent_config
                    )
                    server_runner.run_as_cloud_agent = self.run_as_cloud_agent
                    server_runner.edge_id = edge_id if server_id is None else server_id
                    server_runner.run_status = status
                    server_runner.client_mqtt_mgr = self.client_mqtt_mgr
                    server_runner.mlops_metrics = self.mlops_metrics
                    server_runner.cleanup_client_with_status()
                else:
                    ServerConstants.cleanup_run_process(run_id)

                # Stop log processor for current run
                MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, edge_id)

                time.sleep(1)
                MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, server_id)
            elif self.run_as_cloud_server:
                # Stop log processor for current run
                MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, self.edge_id)
                pass

            if self.run_process_map.get(run_id_str, None) is not None:
                self.run_process_map.pop(run_id_str)

    def cleanup_client_with_status(self):
        if self.run_status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED:
            #logging.info("received to finished status.")
            self.cleanup_run_when_finished(should_send_server_id_status=False)
        elif self.run_status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED:
            #logging.info("received to failed status.")
            self.cleanup_run_when_starting_failed(should_send_server_id_status=False)

    def callback_report_current_status(self, topic, payload):
        logging.info(
            f"FedMLDebug - Receive: topic ({topic}), payload ({payload})"
        )

        request_json = json.loads(payload)
        if self.run_as_edge_server_and_agent:
            self.send_agent_active_msg()
        elif self.run_as_cloud_agent:
            self.send_agent_active_msg()
        elif self.run_as_cloud_server:
            pass

    @staticmethod
    def process_ota_upgrade_msg():
        os.system("pip install -U fedml")

    def callback_server_ota_msg(self, topic, payload):
        logging.info(
            f"FedMLDebug - Receive: topic ({topic}), payload ({payload})"
        )

        request_json = json.loads(payload)
        cmd = request_json["cmd"]

        if cmd == ServerConstants.FEDML_OTA_CMD_UPGRADE:
            try:
                self.process_ota_upgrade_msg()
                # Process(target=FedMLServerRunner.process_ota_upgrade_msg).start()
                raise Exception("After upgraded, restart runner...")
            except Exception as e:
                pass
        elif cmd == ServerConstants.FEDML_OTA_CMD_RESTART:
            raise Exception("Restart runner...")

    def callback_response_device_info(self, topic, payload):
        # Parse payload
        payload_jsonn = json.loads(payload)
        run_id = payload_jsonn.get("run_id", 0)
        master_device_id = payload_jsonn.get("master_device_id", 0)
        slave_device_id = payload_jsonn.get("slave_device_id", 0)
        edge_id = payload_jsonn.get("edge_id", 0)
        device_info = payload_jsonn.get("edge_info", 0)
        device_info["master_device_id"] = master_device_id
        device_info["slave_device_id"] = slave_device_id
        run_id_str = str(run_id)

        # Put device info into a multiprocessing queue so master runner checks if all edges are ready
        if self.edge_status_queue_map.get(run_id_str, None) is None:
            self.edge_status_queue_map[run_id_str] = Queue()
        self.edge_status_queue_map[run_id_str].put(device_info)

        self.check_model_device_ready_and_deploy(run_id, master_device_id, slave_device_id)

    def check_model_device_ready_and_deploy(self, run_id, master_device_id, slave_device_id):
        request_json = self.running_request_json.get(str(run_id), None)
        if request_json is None:
            return
        run_config = self.request_json["run_config"]
        run_params = run_config.get("parameters", {})
        job_yaml = run_params.get("job_yaml", {})
        job_type = job_yaml.get("job_type", None)
        job_type = job_yaml.get("task_type", Constants.JOB_TASK_TYPE_TRAIN) if job_type is None else job_type
        if job_type != Constants.JOB_TASK_TYPE_DEPLOY and job_type != Constants.JOB_TASK_TYPE_SERVE:
            return

        # Init model device ids for each run
        run_id_str = str(run_id)
        if self.run_model_device_ids.get(run_id_str, None) is None:
            self.run_model_device_ids[run_id_str] = list()

        # Append master device and slave devices to the model devices map
        self.run_model_device_ids[run_id_str].append({"master_device_id": master_device_id,
                                                     "slave_device_id": slave_device_id})
        model_device_ids = self.run_model_device_ids.get(run_id_str, None)
        if model_device_ids is None:
            return

        # Check if all model devices are ready
        if len(model_device_ids) != len(self.run_edge_ids.get(run_id_str, list())):
            return

        # Generate model master ids and model slave device ids
        device_master_ids = list()
        device_slave_ids = list()
        for device_ids in model_device_ids:
            model_master_id = device_ids.get("master_device_id")
            model_slave_id = device_ids.get("slave_device_id")
            device_master_ids.append(model_master_id)
            device_slave_ids.append(model_slave_id)

        if len(device_master_ids) <= 0:
            return

        # Generate serving devices for deploying
        serving_devices = list()
        serving_devices.append(device_master_ids[0])
        if len(device_slave_ids) > 1:
            device_slave_ids.pop(0)
        serving_devices.extend(device_slave_ids)

        # Start to deploy the model
        self.deploy_model(serving_devices, request_json)

    @staticmethod
    def get_device_id():
        device_file_path = os.path.join(ServerConstants.get_data_dir(), ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME)
        file_for_device_id = os.path.join(device_file_path, "devices.id")
        if not os.path.exists(device_file_path):
            os.makedirs(device_file_path)
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
                device_id = hex(uuid.getnode())
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
                        pass
                    return str(guid)

                device_id = str(get_uuid())
            elif "posix" in os.name:
                device_id = sys_utils.get_device_id_in_docker()
                if device_id is None:
                    device_id = hex(uuid.getnode())
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

    def bind_account_and_device_id(self, url, account_id, device_id, os_name, api_key="", role=None):
        if role is None:
            role = "edge_server"
            if self.run_as_edge_server_and_agent:
                role = "edge_server"
            elif self.run_as_cloud_agent:
                role = "cloud_agent"
            elif self.run_as_cloud_server:
                role = "cloud_server"

        ip = requests.get('https://checkip.amazonaws.com').text.strip()
        fedml_ver, exec_path, os_ver, cpu_info, python_ver, torch_ver, mpi_installed, \
            cpu_usage, available_mem, total_mem, gpu_info, gpu_available_mem, gpu_total_mem, \
            gpu_count, gpu_vendor, cpu_count, gpu_device_name = get_sys_runner_info()
        host_name = sys_utils.get_host_name()
        json_params = {
            "accountid": account_id,
            "deviceid": device_id,
            "type": os_name,
            "status": ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE,
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

        _, cert_path = MLOpsConfigs.get_instance(self.args).get_request_params()
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    url, json=json_params, verify=True,
                    headers={"content-type": "application/json", "Connection": "close"}
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    url, json=json_params, verify=True,
                    headers={"content-type": "application/json", "Connection": "close"}
                )
        else:
            response = requests.post(url, json=json_params, headers={"Connection": "close"})
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
                if edge_id is None or edge_id <= 0:
                    print(f"Binding to MLOps with response.status_code = {response.status_code}, "
                          f"response.content: {response.content}")
            else:
                print(f"Binding to MLOps with response.status_code = {response.status_code}, "
                      f"response.content: {response.content}")
                return 0, None, None
        return edge_id, user_name, extra_url

    def fetch_configs(self):
        return MLOpsConfigs.get_instance(self.args).fetch_all_configs()

    def send_agent_active_msg(self):
        active_topic = "flserver_agent/active"
        status = MLOpsStatus.get_instance().get_server_agent_status(self.edge_id)
        if (
                status is not None
                and status != ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE
                and status != ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE
        ):
            return

        if self.run_as_cloud_agent:
            status = ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE
        else:
            try:
                current_job = FedMLServerDataInterface.get_instance().get_job_by_id(self.run_id)
            except Exception as e:
                current_job = None
            if current_job is None:
                if status is not None and status == ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE:
                    status = ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE
                else:
                    return
            else:
                status = ServerConstants.get_device_state_from_run_edge_state(current_job.status)
        active_msg = {"ID": self.edge_id, "status": status}
        MLOpsStatus.get_instance().set_server_agent_status(self.edge_id, status)
        self.mqtt_mgr.send_message_json(active_topic, json.dumps(active_msg))

    def recover_start_train_msg_after_upgrading(self):
        try:
            current_job = FedMLServerDataInterface.get_instance().get_current_job()
            if current_job is not None and \
                    current_job.status == ServerConstants.MSG_MLOPS_SERVER_STATUS_UPGRADING:
                logging.info("start training after upgrading.")
                server_agent_id = self.edge_id
                topic_start_train = "mlops/flserver_agent_" + str(server_agent_id) + "/start_train"
                self.callback_start_train(topic_start_train, current_job.running_json)
        except Exception as e:
            logging.info("recover starting train message after upgrading: {}".format(traceback.format_exc()))

    def on_agent_mqtt_connected(self, mqtt_client_object):
        # The MQTT message topic format is as follows: <sender>/<receiver>/<action>

        # Setup MQTT message listener for starting training
        server_agent_id = self.edge_id
        topic_start_train = "mlops/flserver_agent_" + str(server_agent_id) + "/start_train"
        self.mqtt_mgr.add_message_listener(topic_start_train, self.callback_start_train)

        # Setup MQTT message listener for stopping training
        topic_stop_train = "mlops/flserver_agent_" + str(server_agent_id) + "/stop_train"
        self.mqtt_mgr.add_message_listener(topic_stop_train, self.callback_stop_train)

        # Setup MQTT message listener for server status switching
        topic_server_status = "fl_server/flserver_agent_" + str(server_agent_id) + "/status"
        self.mqtt_mgr.add_message_listener(topic_server_status, self.callback_runner_id_status)

        # Setup MQTT message listener to report current device status.
        topic_report_status = "mlops/report_device_status"
        self.mqtt_mgr.add_message_listener(topic_report_status, self.callback_report_current_status)

        # Setup MQTT message listener to OTA messages from the MLOps.
        topic_ota_msg = "mlops/flserver_agent_" + str(server_agent_id) + "/ota"
        self.mqtt_mgr.add_message_listener(topic_ota_msg, self.callback_server_ota_msg)

        # Setup MQTT message listener for running failed
        topic_exit_train_with_exception = "flserver_agent/" + str(server_agent_id) + "/exit_train_with_exception"
        self.mqtt_mgr.add_message_listener(topic_exit_train_with_exception, self.callback_exit_train_with_exception)

        # Setup MQTT message listener to request device info from the client.
        topic_response_device_info = "client/server/response_device_info/" + str(self.edge_id)
        self.mqtt_mgr.add_message_listener(topic_response_device_info, self.callback_response_device_info)

        # Subscribe topics for starting train, stopping train and fetching client status.
        mqtt_client_object.subscribe(topic_start_train, qos=2)
        mqtt_client_object.subscribe(topic_stop_train, qos=2)
        mqtt_client_object.subscribe(topic_server_status, qos=2)
        mqtt_client_object.subscribe(topic_report_status, qos=2)
        mqtt_client_object.subscribe(topic_ota_msg, qos=2)
        mqtt_client_object.subscribe(topic_exit_train_with_exception, qos=2)
        mqtt_client_object.subscribe(topic_response_device_info, qos=2)

        # Broadcast the first active message.
        self.send_agent_active_msg()

        if self.run_as_cloud_server:
            # Start the FedML server
            self.callback_start_train(payload=self.args.runner_cmd)

        # Echo results
        print("\n\nCongratulations, your device is connected to the FedML MLOps platform successfully!")
        print(
            "Your FedML Edge ID is " + str(self.edge_id) + ", unique device ID is "
            + str(self.unique_device_id)
            + "\n"
        )

    def on_agent_mqtt_disconnected(self, mqtt_client_object):
        MLOpsStatus.get_instance().set_server_agent_status(
            self.edge_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE
        )

    def setup_agent_mqtt_connection(self, service_config):
        # Setup MQTT connection
        self.mqtt_mgr = MqttManager(
            service_config["mqtt_config"]["BROKER_HOST"],
            service_config["mqtt_config"]["BROKER_PORT"],
            service_config["mqtt_config"]["MQTT_USER"],
            service_config["mqtt_config"]["MQTT_PWD"],
            service_config["mqtt_config"]["MQTT_KEEPALIVE"],
            "FedML_ServerAgent_Daemon_" + self.args.current_device_id,
            "flserver_agent/last_will_msg",
            json.dumps({"ID": self.edge_id, "status": ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE}),
        )

        # Init local database
        FedMLServerDataInterface.get_instance().create_job_table()

        # Start local API services
        python_program = get_python_program()
        self.local_api_process = ServerConstants.exec_console_with_script(
            "{} -m uvicorn fedml.computing.scheduler.master.server_api:api --host 0.0.0.0 --port {} "
            "--log-level critical".format(python_program, ServerConstants.LOCAL_SERVER_API_PORT),
            should_capture_stdout=False,
            should_capture_stderr=False
        )
        # if self.local_api_process is not None and self.local_api_process.pid is not None:
        #     print(f"Server local API process id {self.local_api_process.pid}")

        # Setup MQTT connected listener
        self.mqtt_mgr.add_connected_listener(self.on_agent_mqtt_connected)
        self.mqtt_mgr.add_disconnected_listener(self.on_agent_mqtt_disconnected)
        self.mqtt_mgr.connect()

        self.setup_client_mqtt_mgr()
        self.mlops_metrics.report_server_training_status(self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE)
        MLOpsStatus.get_instance().set_server_agent_status(
            self.edge_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE
        )

        # MLOpsRuntimeLogDaemon.get_instance(self.args).stop_all_log_processor()

        self.mlops_metrics.stop_device_realtime_perf()
        self.mlops_metrics.report_device_realtime_perf(self.args, service_config["mqtt_config"], is_client=False)

        if not self.run_as_cloud_server:
            self.recover_start_train_msg_after_upgrading()

        JobRunnerUtils.get_instance().sync_run_process_gpu()
        JobRunnerUtils.get_instance().sync_endpoint_process_gpu()
        JobRunnerUtils.get_instance().reset_available_gpu_id_list(self.edge_id)

        # if self.model_device_server is None:
        #     self.model_device_server = FedMLModelDeviceServerRunner(self.args, self.args.current_device_id,
        #                                                             self.args.os_name, self.args.is_from_docker,
        #                                                             self.agent_config)
        #     self.model_device_server.start()

    def start_agent_mqtt_loop(self):
        # Start MQTT message loop
        try:
            self.mqtt_mgr.loop_forever()
        except Exception as e:
            if str(e) == "Restarting after upgraded...":
                logging.info("Restarting after upgraded...")
            else:
                logging.info("Server tracing: {}".format(traceback.format_exc()))
            self.mqtt_mgr.loop_stop()
            self.mqtt_mgr.disconnect()
            self.release_client_mqtt_mgr()

            if self.model_device_server is not None:
                self.model_device_server.stop()

            time.sleep(5)
            sys_utils.cleanup_all_fedml_server_login_processes(
                ServerConstants.SERVER_LOGIN_PROGRAM, clean_process_group=False)
            sys.exit(1)
