import json
import logging
import multiprocessing
import sys

from multiprocessing import Process
import os
import platform
import shutil
import stat
import subprocess
import threading

import time
import traceback
import urllib
import uuid
import zipfile

import requests
from ...core.mlops.mlops_runtime_log import MLOpsRuntimeLog

from ...core.distributed.communication.mqtt.mqtt_manager import MqttManager
from ...cli.comm_utils.yaml_utils import load_yaml_config
from ...cli.edge_deployment.client_constants import ClientConstants

from ...core.mlops.mlops_metrics import MLOpsMetrics

from ...core.mlops.mlops_configs import MLOpsConfigs
from ...core.mlops.mlops_runtime_log_daemon import MLOpsRuntimeLogDaemon
from ...core.mlops.mlops_status import MLOpsStatus
from ..comm_utils.sys_utils import get_sys_runner_info, get_python_program
from .client_data_interface import FedMLClientDataInterface
from ..comm_utils import sys_utils
from ...core.mlops.mlops_utils import MLOpsUtils


class RunnerError(Exception):
    """ Runner failed. """
    pass


class FedMLClientRunner:

    def __init__(self, args, edge_id=0, request_json=None, agent_config=None, run_id=0):
        self.run_process_event = None
        self.run_process = None
        self.local_api_process = None
        self.start_request_json = None
        self.device_status = None
        self.current_training_status = None
        self.mqtt_mgr = None
        self.client_mqtt_mgr = None
        self.client_mqtt_is_connected = False
        self.client_mqtt_lock = None
        self.edge_id = edge_id
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
        # logging.info("Current directory of client agent: " + self.cur_dir)

    def build_dynamic_constrain_variables(self, run_id, run_config):
        data_config = run_config["data_config"]
        server_edge_id_list = self.request_json["edgeids"]
        local_edge_id_list = [1]
        local_edge_id_list[0] = self.edge_id
        is_using_local_data = 0
        private_data_dir = data_config["privateLocalData"]
        synthetic_data_url = data_config["syntheticDataUrl"]
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
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.CLIENT_INDEX}"] = server_edge_id_list.index(self.edge_id) + 1
        client_objects = str(json.dumps(edges))
        client_objects = client_objects.replace(" ", "").replace("\n", "").replace('"', '\\"')
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.CLIENT_OBJECT_LIST}"] = client_objects
        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES["${FEDSYS.LOG_SERVER_URL}"] = self.agent_config["ml_ops_config"][
            "LOG_SERVER_URL"
        ]

    def unzip_file(self, zip_file, unzip_file_path):
        result = False
        if zipfile.is_zipfile(zip_file):
            with zipfile.ZipFile(zip_file, "r") as zipf:
                zipf.extractall(unzip_file_path)
                result = True

        return result

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
        local_package_file = os.path.join(local_package_path, os.path.basename(package_url))
        if os.path.exists(local_package_file):
            os.remove(local_package_file)
        urllib.request.urlretrieve(package_url, local_package_file, reporthook=self.package_download_progress)
        unzip_package_path = ClientConstants.get_package_unzip_dir()
        try:
            shutil.rmtree(ClientConstants.get_package_run_dir(package_name), ignore_errors=True)
        except Exception as e:
            pass
        logging.info("local_package_file {}, unzip_package_path {}, unzip file full path {}".format(
            local_package_file, unzip_package_path, ClientConstants.get_package_run_dir(package_name)))
        self.unzip_file(local_package_file, unzip_package_path)
        unzip_package_path = ClientConstants.get_package_run_dir(package_name)
        return unzip_package_path

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
        if not self.build_dynamic_args(run_id, run_config, package_conf_object, unzip_package_path):
            return None, None

        return unzip_package_path, package_conf_object

    def build_dynamic_args(self, run_id, run_config, package_conf_object, base_dir):
        fedml_conf_file = package_conf_object["entry_config"]["conf_file"]
        fedml_conf_file_processed = str(fedml_conf_file).replace('\\', os.sep).replace('/', os.sep)
        fedml_conf_path = os.path.join(base_dir, "fedml", "config",
                                       os.path.basename(fedml_conf_file_processed))
        fedml_conf_object = load_yaml_config(fedml_conf_path)

        # Replace local fedml config objects with parameters from MLOps web
        parameters_object = run_config.get("parameters", None)
        if parameters_object is not None:
            fedml_conf_object = parameters_object

        package_dynamic_args = package_conf_object["dynamic_args"]
        fedml_conf_object["comm_args"]["mqtt_config_path"] = package_dynamic_args["mqtt_config_path"]
        fedml_conf_object["comm_args"]["s3_config_path"] = package_dynamic_args["s3_config_path"]
        fedml_conf_object["common_args"]["using_mlops"] = True
        fedml_conf_object["train_args"]["run_id"] = package_dynamic_args["run_id"]
        fedml_conf_object["train_args"]["client_id_list"] = package_dynamic_args["client_id_list"]
        fedml_conf_object["train_args"]["client_num_in_total"] = int(package_dynamic_args["client_num_in_total"])
        fedml_conf_object["train_args"]["client_num_per_round"] = int(package_dynamic_args["client_num_in_total"])
        fedml_conf_object["train_args"]["client_id"] = self.edge_id
        fedml_conf_object["train_args"]["server_id"] = self.request_json.get("server_id", "0")
        fedml_conf_object["device_args"]["worker_num"] = int(package_dynamic_args["client_num_in_total"])
        # fedml_conf_object["data_args"]["data_cache_dir"] = package_dynamic_args["data_cache_dir"]
        fedml_conf_object["tracking_args"]["log_file_dir"] = package_dynamic_args["log_file_dir"]
        fedml_conf_object["tracking_args"]["log_server_url"] = package_dynamic_args["log_server_url"]
        if hasattr(self.args, "local_server") and self.args.local_server is not None:
            fedml_conf_object["comm_args"]["local_server"] = self.args.local_server
        bootstrap_script_path = None
        env_args = fedml_conf_object.get("environment_args", None)
        if env_args is not None:
            bootstrap_script_file = env_args.get("bootstrap", None)
            if bootstrap_script_file is not None:
                bootstrap_script_file = str(bootstrap_script_file).replace('\\', os.sep).replace('/', os.sep)
                if platform.system() == 'Windows':
                    bootstrap_script_file = bootstrap_script_file.replace('.sh', '.bat')
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

        ClientConstants.generate_yaml_doc(fedml_conf_object, fedml_conf_path)

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
                    process = ClientConstants.exec_console_with_script(bootstrap_scripts, should_capture_stdout=True,
                                                                       should_capture_stderr=True)
                    ClientConstants.save_bootstrap_process(run_id, process.pid)
                    ret_code, out, err = ClientConstants.get_console_pipe_out_err_results(process)
                    if ret_code is None or ret_code <= 0:
                        if out is not None:
                            out_str = out.decode(encoding="utf-8")
                            if out_str != "":
                                logging.info("{}".format(out_str))

                        sys_utils.log_return_info(bootstrap_script_file, 0)

                        is_bootstrap_run_ok = True
                    else:
                        if err is not None:
                            err_str = err.decode(encoding="utf-8")
                            if err_str != "":
                                logging.error("{}".format(err_str))

                        sys_utils.log_return_info(bootstrap_script_file, ret_code)

                        is_bootstrap_run_ok = False
        except Exception as e:
            logging.error("Bootstrap script error: {}".format(traceback.format_exc()))
            is_bootstrap_run_ok = False

        return is_bootstrap_run_ok

    def run(self, process_event):
        os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
        os.environ.setdefault('PYTHONWARNINGS', 'ignore:semaphore_tracker:UserWarning')

        self.run_process_event = process_event
        try:
            MLOpsUtils.set_ntp_offset(self.ntp_offset)
            self.setup_client_mqtt_mgr()
            self.run_impl()
        except RunnerError:
            logging.info("Runner stopped.")
            self.reset_devices_status(self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED)
        except Exception as e:
            logging.error("Runner exits with exceptions. {}".format(traceback.format_exc()))
            self.reset_devices_status(self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED)
        finally:
            logging.info("Release resources.")
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(self.run_id, self.edge_id)
            if self.mlops_metrics is not None:
                self.mlops_metrics.stop_sys_perf()
            time.sleep(3)
            ClientConstants.cleanup_learning_process(self.run_id)
            ClientConstants.cleanup_run_process(self.run_id)
            self.release_client_mqtt_mgr()

    def check_runner_stop_event(self):
        if self.run_process_event.is_set():
            logging.info("Received stopping event.")
            raise RunnerError("Runner stopped")

    def run_impl(self):
        run_id = self.request_json["runId"]
        run_config = self.request_json["run_config"]
        data_config = run_config["data_config"]
        packages_config = run_config["packages_config"]

        self.check_runner_stop_event()

        MLOpsRuntimeLog.get_instance(self.args).init_logs(show_stdout_log=True)

        self.mlops_metrics.report_client_training_status(self.edge_id,
                                                         ClientConstants.MSG_MLOPS_CLIENT_STATUS_INITIALIZING,
                                                         self.start_request_json,
                                                         in_run_id=run_id)

        # get training params
        private_local_data_dir = data_config.get("privateLocalData", "")
        is_using_local_data = 0
        # if private_local_data_dir is not None and len(str(private_local_data_dir).strip(' ')) > 0:
        #     is_using_local_data = 1

        # start a run according to the hyper-parameters
        # fedml_local_data_dir = self.cur_dir + "/fedml_data/run_" + str(run_id) + "_edge_" + str(edge_id)
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
            self.mlops_metrics.client_send_exit_train_msg(run_id, self.edge_id,
                                                          ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED)
            return

        logging.info("cleanup the previous learning process and check downloaded packages...")

        entry_file_config = fedml_config_object["entry_config"]
        dynamic_args_config = fedml_config_object["dynamic_args"]
        entry_file = str(entry_file_config["entry_file"]).replace('\\', os.sep).replace('/', os.sep)
        entry_file = os.path.basename(entry_file)
        conf_file = entry_file_config["conf_file"]
        conf_file = str(conf_file).replace('\\', os.sep).replace('/', os.sep)
        ClientConstants.cleanup_learning_process(run_id)
        ClientConstants.cleanup_bootstrap_process(run_id)
        if not os.path.exists(unzip_package_path):
            logging.info("failed to unzip file.")
            self.check_runner_stop_event()
            self.cleanup_run_when_starting_failed()
            self.mlops_metrics.client_send_exit_train_msg(run_id, self.edge_id,
                                                          ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED)
            return
        os.chdir(os.path.join(unzip_package_path, "fedml"))

        self.check_runner_stop_event()

        logging.info("starting the learning process...")

        python_program = get_python_program()
        entry_fill_full_path = os.path.join(unzip_package_path, "fedml", entry_file)
        conf_file_full_path = os.path.join(unzip_package_path, "fedml", conf_file)
        logging.info("Run the client: {} {} --cf {} --rank {} --role client".format(
            python_program, entry_fill_full_path, conf_file_full_path, str(dynamic_args_config["rank"])))
        process = ClientConstants.exec_console_with_shell_script_list(
            [
                python_program,
                entry_fill_full_path,
                "--cf",
                conf_file_full_path,
                "--rank",
                str(dynamic_args_config["rank"]),
                "--role",
                "client"
            ],
            should_capture_stdout=False,
            should_capture_stderr=True
        )
        logging.info("waiting the learning process to train models...")
        ClientConstants.save_learning_process(run_id, process.pid)

        ret_code, out, err = ClientConstants.get_console_pipe_out_err_results(process)
        is_run_ok = sys_utils.is_runner_finished_normally(process.pid)
        if ret_code is None or ret_code <= 0:
            if is_run_ok:
                if out is not None:
                    out_str = out.decode(encoding="utf-8")
                    if out_str != "":
                        logging.info("{}".format(out_str))

                sys_utils.log_return_info(entry_file, 0)
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
                err_str = err.decode(encoding="utf-8")
                if err_str != "":
                    logging.error("{}".format(err_str))

            sys_utils.log_return_info(entry_file, ret_code)

            self.mlops_metrics.report_client_id_status(run_id, self.edge_id,
                                                       ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED)

            self.mlops_metrics.client_send_exit_train_msg(run_id, self.edge_id,
                                                          ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED)

    def reset_devices_status(self, edge_id, status):
        self.mlops_metrics.run_id = self.run_id
        self.mlops_metrics.edge_id = edge_id
        self.mlops_metrics.broadcast_client_training_status(edge_id, status)

    def stop_run(self):
        logging.info("Stop run successfully.")

        self.reset_devices_status(self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED)

        try:
            ClientConstants.cleanup_learning_process(self.run_id)
            ClientConstants.cleanup_bootstrap_process(self.run_id)
            ClientConstants.cleanup_run_process(self.run_id)
        except Exception as e:
            pass

    def stop_run_entry(self):
        try:
            if self.run_process_event is not None:
                self.run_process_event.set()
            self.stop_run_with_killed_status()
            # if self.run_process is not None:
            #     logging.info("Run will be stopped, waiting...")
            #     self.run_process.join()
        except Exception as e:
            ClientConstants.cleanup_run_process(self.run_id)

    def stop_run_with_killed_status(self, report_status=True):
        # logging.info("Stop run successfully.")
        try:
            ClientConstants.cleanup_learning_process(self.run_id)
            ClientConstants.cleanup_bootstrap_process(self.run_id)
        except Exception as e:
            pass

        if report_status:
            self.reset_devices_status(self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED)

    def exit_run_with_exception_entry(self):
        try:
            self.setup_client_mqtt_mgr()
            self.exit_run_with_exception()
        except Exception as e:
            self.release_client_mqtt_mgr()
        finally:
            self.release_client_mqtt_mgr()

    def exit_run_with_exception(self):
        logging.info("Exit run successfully.")

        ClientConstants.cleanup_learning_process(self.run_id)
        ClientConstants.cleanup_run_process(self.run_id)
        ClientConstants.cleanup_bootstrap_process(self.run_id)

        self.mlops_metrics.report_client_id_status(self.run_id, self.edge_id,
                                                   ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED)

        time.sleep(1)

    def cleanup_run_when_starting_failed(self):
        logging.error("Cleanup run successfully when starting failed.")

        self.reset_devices_status(self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED)

        time.sleep(2)

        try:
            self.mlops_metrics.stop_sys_perf()
        except Exception as ex:
            pass

        time.sleep(1)

        try:
            ClientConstants.cleanup_learning_process(self.run_id)
            ClientConstants.cleanup_bootstrap_process(self.run_id)
        except Exception as e:
            pass

    def cleanup_run_when_finished(self):
        logging.info("Cleanup run successfully when finished.")

        self.reset_devices_status(self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED)

        time.sleep(2)

        try:
            self.mlops_metrics.stop_sys_perf()
        except Exception as ex:
            pass

        time.sleep(1)

        try:
            ClientConstants.cleanup_learning_process(self.run_id)
            ClientConstants.cleanup_bootstrap_process(self.run_id)
        except Exception as e:
            pass

    def on_client_mqtt_disconnected(self, mqtt_client_object):
        if self.client_mqtt_lock is None:
            self.client_mqtt_lock = threading.Lock()

        self.client_mqtt_lock.acquire()
        self.client_mqtt_is_connected = False
        self.client_mqtt_lock.release()

    def on_client_mqtt_connected(self, mqtt_client_object):
        if self.mlops_metrics is None:
            self.mlops_metrics = MLOpsMetrics()

        self.mlops_metrics.set_messenger(self.client_mqtt_mgr)
        self.mlops_metrics.run_id = self.run_id

        if self.client_mqtt_lock is None:
            self.client_mqtt_lock = threading.Lock()

        self.client_mqtt_lock.acquire()
        self.client_mqtt_is_connected = True
        self.client_mqtt_lock.release()

    def setup_client_mqtt_mgr(self):
        if self.client_mqtt_mgr is not None:
            return

        if self.client_mqtt_lock is None:
            self.client_mqtt_lock = threading.Lock()

        self.client_mqtt_mgr = MqttManager(
            self.agent_config["mqtt_config"]["BROKER_HOST"],
            self.agent_config["mqtt_config"]["BROKER_PORT"],
            self.agent_config["mqtt_config"]["MQTT_USER"],
            self.agent_config["mqtt_config"]["MQTT_PWD"],
            self.agent_config["mqtt_config"]["MQTT_KEEPALIVE"],
            "FedML_ClientAgent_Metrics_{}_{}_{}".format(self.args.current_device_id,
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

    def ota_upgrade(self, payload, request_json):
        no_upgrade = False
        upgrade_version = None
        run_id = request_json["runId"]

        try:
            run_config = request_json.get("run_config", None)
            parameters = run_config.get("parameters", None)
            common_args = parameters.get("common_args", None)
            no_upgrade = common_args.get("no_upgrade", False)
            upgrade_version = common_args.get("upgrade_version", None)
        except Exception as e:
            pass

        should_upgrade = True
        if upgrade_version is None or upgrade_version == "latest":
            try:
                fedml_is_latest_version, local_ver, remote_ver = sys_utils. \
                    check_fedml_is_latest_version(self.version)
            except Exception as e:
                return

            if fedml_is_latest_version:
                should_upgrade = False
            upgrade_version = remote_ver

        if no_upgrade:
            should_upgrade = False

        if should_upgrade:
            FedMLClientDataInterface.get_instance(). \
                save_started_job(run_id, self.edge_id, time.time(),
                                 ClientConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING,
                                 ClientConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING,
                                 payload)
            self.mlops_metrics. \
                report_client_training_status(self.edge_id,
                                              ClientConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING,
                                              in_run_id=run_id)

            logging.info(f"Upgrade to version {upgrade_version} ...")

            sys_utils.do_upgrade(self.version, upgrade_version)

            raise Exception("Restarting after upgraded...")

    def callback_start_train(self, topic, payload):
        try:
            _, _ = MLOpsConfigs.get_instance(self.args).fetch_configs()
        except Exception as e:
            pass

        if not FedMLClientDataInterface.get_instance().get_agent_status():
            request_json = json.loads(payload)
            run_id = request_json["runId"]
            logging.error(
                "FedMLDebug - Receive: topic ({}), payload ({}), but the client agent is disabled. {}".format(
                    topic, payload, traceback.format_exc()
                )
            )
            self.mlops_metrics.client_send_exit_train_msg(run_id, self.edge_id,
                                                          ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED,
                                                          msg=f"the client agent {self.edge_id} is disabled")
            self.mlops_metrics.report_client_training_status(self.edge_id,
                                                             ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED,
                                                             in_run_id=run_id)
            return

        logging.info(
            f"FedMLDebug - Receive: topic ({topic}), payload ({payload})"
        )

        # Get training params
        request_json = json.loads(payload)
        is_retain = request_json.get("is_retain", False)
        if is_retain:
            return
        self.start_request_json = payload
        run_id = request_json["runId"]

        logging.info("cleanup and save runner information")

        # Terminate previous process about starting or stopping run command
        server_agent_id = request_json["cloud_agent_id"]
        ClientConstants.cleanup_run_process(run_id)
        ClientConstants.save_runner_infos(self.args.device_id + "." + self.args.os_name, self.edge_id, run_id=run_id)

        # Start log processor for current run
        self.args.run_id = run_id
        self.args.edge_id = self.edge_id
        MLOpsRuntimeLog.get_instance(self.args).init_logs(show_stdout_log=True)
        logging.info("start the log processor")
        MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(run_id, self.edge_id)

        self.ota_upgrade(payload, request_json)

        # Start server with multiprocessing mode
        self.request_json = request_json
        client_runner = FedMLClientRunner(
            self.args, edge_id=self.edge_id, request_json=request_json, agent_config=self.agent_config, run_id=run_id
        )
        client_runner.start_request_json = self.start_request_json
        if self.run_process_event is None:
            self.run_process_event = multiprocessing.Event()
        self.run_process_event.clear()
        client_runner.run_process_event = self.run_process_event
        logging.info("start the runner process.")
        self.run_process = Process(target=client_runner.run, args=(self.run_process_event,))
        self.run_process.start()
        ClientConstants.save_run_process(run_id, self.run_process.pid)

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

        # logging.info("Stop run with multiprocessing...")

        # Stop client with multiprocessing mode
        self.request_json = request_json
        client_runner = FedMLClientRunner(
            self.args, edge_id=self.edge_id, request_json=request_json, agent_config=self.agent_config, run_id=run_id
        )
        client_runner.run_process_event = self.run_process_event
        client_runner.run_process = self.run_process
        client_runner.client_mqtt_mgr = self.client_mqtt_mgr
        client_runner.mlops_metrics = self.mlops_metrics
        client_runner.stop_run_entry()

        # Stop log processor for current run
        MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, self.edge_id)

    def callback_exit_train_with_exception(self, topic, payload):
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

        # Stop client with multiprocessing mode
        self.request_json = request_json
        client_runner = FedMLClientRunner(
            self.args, edge_id=self.edge_id, request_json=request_json, agent_config=self.agent_config, run_id=run_id
        )
        try:
            Process(target=client_runner.exit_run_with_exception_entry).start()
        except Exception as e:
            pass

    def cleanup_client_with_status(self):
        if self.device_status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED:
            logging.info("received to finished status.")
            self.cleanup_run_when_finished()
        elif self.device_status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED:
            logging.error("received to failed status from the server agent")
            self.cleanup_run_when_starting_failed()

    def callback_runner_id_status(self, topic, payload):
        # logging.info("callback_runner_id_status: topic = %s, payload = %s" % (topic, payload))
        logging.info(
            f"FedMLDebug - Receive: topic ({topic}), payload ({payload})"
        )

        request_json = json.loads(payload)
        is_retain = request_json.get("is_retain", False)
        if is_retain:
            return
        run_id = request_json["run_id"]
        edge_id = request_json["edge_id"]
        status = request_json["status"]

        self.save_training_status(edge_id, status)

        if status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED or \
                status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED:
            # Stop client with multiprocessing mode
            self.request_json = request_json
            client_runner = FedMLClientRunner(
                self.args,
                edge_id=self.edge_id,
                request_json=request_json,
                agent_config=self.agent_config,
                run_id=run_id,
            )
            client_runner.device_status = status
            client_runner.client_mqtt_mgr = self.client_mqtt_mgr
            client_runner.mlops_metrics = self.mlops_metrics
            client_runner.cleanup_client_with_status()

            # Stop log processor for current run
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, edge_id)

    def callback_report_current_status(self, topic, payload):
        logging.info(
            f"FedMLDebug - Receive: topic ({topic}), payload ({payload})"
        )

        self.send_agent_active_msg()

    @staticmethod
    def process_ota_upgrade_msg():
        os.system("pip install -U fedml")

    def callback_client_ota_msg(self, topic, payload):
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

    def save_training_status(self, edge_id, training_status):
        self.current_training_status = training_status
        ClientConstants.save_training_infos(edge_id, training_status)

    @staticmethod
    def get_device_id():
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
                logging.info(device_id)
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

    def bind_account_and_device_id(self, url, account_id, device_id, os_name, role="client"):
        ip = requests.get('https://checkip.amazonaws.com').text.strip()
        fedml_ver, exec_path, os_ver, cpu_info, python_ver, torch_ver, mpi_installed, \
            cpu_usage, available_mem, total_mem, gpu_info, gpu_available_mem, gpu_total_mem = get_sys_runner_info()
        json_params = {
            "accountid": account_id,
            "deviceid": device_id,
            "type": os_name,
            "status": ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE,
            "processor": cpu_info,
            "core_type": cpu_info,
            "network": "",
            "role": role,
            "os_ver": os_ver,
            "memory": total_mem,
            "ip": ip,
            "extra_infos": {"fedml_ver": fedml_ver, "exec_path": exec_path, "os_ver": os_ver,
                            "cpu_info": cpu_info, "python_ver": python_ver, "torch_ver": torch_ver,
                            "mpi_installed": mpi_installed, "cpu_sage": cpu_usage,
                            "available_mem": available_mem, "total_mem": total_mem}
        }
        if gpu_info is not None:
            if gpu_total_mem is not None:
                json_params["gpu"] = gpu_info + ", Total GPU Memory: " + gpu_total_mem
            else:
                json_params["gpu"] = gpu_info
            json_params["extra_infos"]["gpu_info"] = gpu_info
            if gpu_available_mem is not None:
                json_params["extra_infos"]["gpu_available_mem"] = gpu_available_mem
            if gpu_total_mem is not None:
                json_params["extra_infos"]["gpu_total_mem"] = gpu_total_mem
        else:
            json_params["gpu"] = "None"

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
        # print("url = {}, response = {}".format(url, response))
        status_code = response.json().get("code")
        if status_code == "SUCCESS":
            edge_id = response.json().get("data").get("id")
        else:
            return 0
        return edge_id

    def fetch_configs(self):
        return MLOpsConfigs.get_instance(self.args).fetch_all_configs()

    def send_agent_active_msg(self):
        active_topic = "flclient_agent/active"
        status = MLOpsStatus.get_instance().get_client_agent_status(self.edge_id)
        if (
                status is not None
                and status != ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE
                and status != ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE
        ):
            return

        try:
            current_job = FedMLClientDataInterface.get_instance().get_job_by_id(self.run_id)
        except Exception as e:
            current_job = None
        if current_job is None:
            if status is not None and status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE:
                status = ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE
            else:
                return
        else:
            status = ClientConstants.get_device_state_from_run_edge_state(current_job.status)
        active_msg = {"ID": self.edge_id, "status": status}
        MLOpsStatus.get_instance().set_client_agent_status(self.edge_id, status)
        self.mqtt_mgr.send_message_json(active_topic, json.dumps(active_msg))

    def recover_start_train_msg_after_upgrading(self):
        try:
            current_job = FedMLClientDataInterface.get_instance().get_current_job()
            if current_job is not None and \
                    current_job.status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING:
                logging.info("start training after upgrading.")
                topic_start_train = "flserver_agent/" + str(self.edge_id) + "/start_train"
                self.callback_start_train(topic_start_train, current_job.running_json)
        except Exception as e:
            logging.info("recover starting train message after upgrading: {}".format(traceback.format_exc()))

    def on_agent_mqtt_connected(self, mqtt_client_object):
        # The MQTT message topic format is as follows: <sender>/<receiver>/<action>

        # Setup MQTT message listener for starting training
        topic_start_train = "flserver_agent/" + str(self.edge_id) + "/start_train"
        self.mqtt_mgr.add_message_listener(topic_start_train, self.callback_start_train)

        # Setup MQTT message listener for stopping training
        topic_stop_train = "flserver_agent/" + str(self.edge_id) + "/stop_train"
        self.mqtt_mgr.add_message_listener(topic_stop_train, self.callback_stop_train)

        # Setup MQTT message listener for running failed
        topic_exit_train_with_exception = "flserver_agent/" + str(self.edge_id) + "/exit_train_with_exception"
        self.mqtt_mgr.add_message_listener(topic_exit_train_with_exception, self.callback_exit_train_with_exception)

        # Setup MQTT message listener for client status switching
        topic_client_status = "fl_client/flclient_agent_" + str(self.edge_id) + "/status"
        self.mqtt_mgr.add_message_listener(topic_client_status, self.callback_runner_id_status)

        # Setup MQTT message listener to report current device status.
        topic_report_status = "mlops/report_device_status"
        self.mqtt_mgr.add_message_listener(topic_report_status, self.callback_report_current_status)

        # Setup MQTT message listener to OTA messages from the MLOps.
        topic_ota_msg = "mlops/flclient_agent_" + str(self.edge_id) + "/ota"
        self.mqtt_mgr.add_message_listener(topic_ota_msg, self.callback_client_ota_msg)

        # Subscribe topics for starting train, stopping train and fetching client status.
        mqtt_client_object.subscribe(topic_start_train, qos=2)
        mqtt_client_object.subscribe(topic_stop_train, qos=2)
        mqtt_client_object.subscribe(topic_client_status, qos=2)
        mqtt_client_object.subscribe(topic_report_status, qos=2)
        mqtt_client_object.subscribe(topic_exit_train_with_exception, qos=2)
        mqtt_client_object.subscribe(topic_ota_msg, qos=2)

        # Broadcast the first active message.
        self.send_agent_active_msg()

        # Echo results
        print("\n\nCongratulations, your device is connected to the FedML MLOps platform successfully!")
        print(
            "Your FedML Edge ID is " + str(self.edge_id) + ", unique device ID is "
            + str(self.unique_device_id)
            + "\n"
        )

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
            "FedML_ClientAgent_Daemon_" + self.args.current_device_id,
            "flclient_agent/last_will_msg",
            json.dumps({"ID": self.edge_id, "status": ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE}),
        )
        self.agent_config = service_config

        # Init local database
        FedMLClientDataInterface.get_instance().create_job_table()

        # Start local API services
        python_program = get_python_program()
        self.local_api_process = ClientConstants.exec_console_with_script(
            "{} -m uvicorn fedml.cli.edge_deployment.client_api:api --host 0.0.0.0 --port {} "
            "--log-level critical".format(python_program,
                                          ClientConstants.LOCAL_CLIENT_API_PORT),
            should_capture_stdout=False,
            should_capture_stderr=False
        )

        # Setup MQTT connected listener
        self.mqtt_mgr.add_connected_listener(self.on_agent_mqtt_connected)
        self.mqtt_mgr.add_disconnected_listener(self.on_agent_mqtt_disconnected)
        self.mqtt_mgr.connect()

        self.setup_client_mqtt_mgr()
        self.mlops_metrics.report_client_training_status(self.edge_id,
                                                         ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE)
        MLOpsStatus.get_instance().set_client_agent_status(self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE)

        MLOpsRuntimeLogDaemon.get_instance(self.args).stop_all_log_processor()

        self.recover_start_train_msg_after_upgrading()

    def start_agent_mqtt_loop(self):
        # Start MQTT message loop
        try:
            self.mqtt_mgr.loop_forever()
        except Exception as e:
            if str(e) == "Restarting after upgraded...":
                logging.info("Restarting after upgraded...")
            else:
                logging.info("Client tracing: {}".format(traceback.format_exc()))
            self.mqtt_mgr.loop_stop()
            self.mqtt_mgr.disconnect()
            self.release_client_mqtt_mgr()
            time.sleep(5)
            sys_utils.cleanup_all_fedml_client_login_processes(
                ClientConstants.CLIENT_LOGIN_PROGRAM, clean_process_group=False)
            sys.exit(1)
