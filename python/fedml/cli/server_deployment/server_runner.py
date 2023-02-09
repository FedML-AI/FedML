import base64
import copy
import json
import logging
import platform

import multiprocess as multiprocessing
from multiprocessing import Process
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

import click
import requests
from ...core.mlops.mlops_runtime_log import MLOpsRuntimeLog

from ...core.distributed.communication.mqtt.mqtt_manager import MqttManager
from ...cli.comm_utils.yaml_utils import load_yaml_config
from ...cli.edge_deployment.client_constants import ClientConstants
from ...cli.server_deployment.server_constants import ServerConstants

from ...core.mlops.mlops_metrics import MLOpsMetrics

from ...core.mlops.mlops_configs import MLOpsConfigs
from ...core.mlops.mlops_runtime_log_daemon import MLOpsRuntimeLogDaemon
from ...core.mlops.mlops_status import MLOpsStatus
from ..comm_utils.sys_utils import get_sys_runner_info, get_python_program
from ..comm_utils import sys_utils
from .server_data_interface import FedMLServerDataInterface


class FedMLServerRunner:
    FEDML_CLOUD_SERVER_PREFIX = "fedml-server-run-"

    def __init__(self, args, run_id=0, request_json=None, agent_config=None, edge_id=0):
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

    def build_dynamic_constrain_variables(self, run_id, run_config):
        data_config = run_config["data_config"]
        server_edge_id_list = self.request_json["edgeids"]
        is_using_local_data = 0
        private_data_dir = data_config["privateLocalData"]
        synthetic_data_url = data_config["syntheticDataUrl"]
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

    def unzip_file(self, zip_file, unzip_file_path):
        result = False
        if zipfile.is_zipfile(zip_file):
            with zipfile.ZipFile(zip_file, "r") as zipf:
                zipf.extractall(unzip_file_path)
                result = True

        return result

    def retrieve_and_unzip_package(self, package_name, package_url):
        local_package_path = ServerConstants.get_package_download_dir()
        try:
            os.makedirs(local_package_path)
        except Exception as e:
            pass
        local_package_file = os.path.join(local_package_path, os.path.basename(package_url))
        if not os.path.exists(local_package_file):
            urllib.request.urlretrieve(package_url, local_package_file)
        unzip_package_path = ServerConstants.get_package_unzip_dir(self.run_id, package_url)
        self.fedml_packages_base_dir = unzip_package_path
        try:
            shutil.rmtree(
                ServerConstants.get_package_run_dir(self.run_id, package_url, package_name), ignore_errors=True
            )
        except Exception as e:
            pass
        self.unzip_file(local_package_file, unzip_package_path)
        unzip_package_path = ServerConstants.get_package_run_dir(self.run_id, package_url, package_name)
        return unzip_package_path

    def update_local_fedml_config(self, run_id, run_config):
        packages_config = run_config["packages_config"]

        # Copy config file from the client
        unzip_package_path = self.retrieve_and_unzip_package(packages_config["server"], packages_config["serverUrl"])
        self.fedml_packages_unzip_dir = unzip_package_path
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
        log_file_dir = ServerConstants.get_log_file_dir()
        try:
            os.makedirs(log_file_dir)
        except Exception as e:
            pass
        package_conf_object["dynamic_args"]["log_file_dir"] = log_file_dir

        # Save new config dictionary to local file
        fedml_updated_config_file = os.path.join(unzip_package_path, "conf", "fedml.yaml")
        ServerConstants.generate_yaml_doc(package_conf_object, fedml_updated_config_file)

        # Build dynamic arguments and set arguments to fedml config object
        if not self.build_dynamic_args(run_config, package_conf_object, unzip_package_path):
            return None, None

        return unzip_package_path, package_conf_object

    def build_dynamic_args(self, run_config, package_conf_object, base_dir):
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
        fedml_conf_object["train_args"]["server_id"] = self.edge_id
        fedml_conf_object["train_args"]["server_agent_id"] = self.request_json.get("cloud_agent_id", self.edge_id)
        fedml_conf_object["train_args"]["group_server_id_list"] = self.request_json.get("group_server_id_list", list())
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
            bootstrap_script_file = str(bootstrap_script_file).replace('\\', os.sep).replace('/', os.sep)
            if platform.system() == 'Windows':
                bootstrap_script_file = bootstrap_script_file.replace('.sh', '.bat')
            if bootstrap_script_file is not None:
                bootstrap_script_dir = os.path.join(base_dir, "fedml", os.path.dirname(bootstrap_script_file))
                bootstrap_script_path = os.path.join(
                    bootstrap_script_dir, bootstrap_script_dir, os.path.basename(bootstrap_script_file)
                )
        # try:
        #     os.makedirs(package_dynamic_args["data_cache_dir"])
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
                    process = ServerConstants.exec_console_with_script(bootstrap_scripts,
                                                                       should_capture_stdout=True,
                                                                       should_capture_stderr=True)
                    ret_code, out, err = ServerConstants.get_console_pipe_out_err_results(process)
                    if ret_code is None or ret_code == 0:
                        out_str = out.decode(encoding="utf-8")
                        if out_str != "":
                            logging.info("{}".format(out_str))

                        sys_utils.log_return_info(bootstrap_script_file, ret_code)

                        is_bootstrap_run_ok = True
                    else:
                        if err is not None:
                            err_str = err.decode(encoding="utf-8")
                            if err_str != "":
                                logging.error("{}".format(err_str))

                        sys_utils.log_return_info(bootstrap_script_file, ret_code)

                        is_bootstrap_run_ok = False
        except Exception as e:
            logging.error("Bootstrap scripts error: {}".format(traceback.format_exc()))

            is_bootstrap_run_ok = False

        return is_bootstrap_run_ok

    def run(self):
        run_id = self.request_json["runId"]
        run_config = self.request_json["run_config"]
        data_config = run_config["data_config"]
        packages_config = run_config["packages_config"]
        self.run_id = run_id

        self.args.run_id = self.run_id
        MLOpsRuntimeLog.get_instance(self.args).init_logs(show_stdout_log=True)

        # set mqtt connection for client
        self.setup_client_mqtt_mgr()
        self.wait_client_mqtt_connected()
        self.send_training_request_to_edges()
        self.release_client_mqtt_mgr()

        # report server running status
        self.mlops_metrics.report_server_training_status(run_id,
                                                         ServerConstants.MSG_MLOPS_SERVER_STATUS_STARTING,
                                                         running_json=self.start_request_json)

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

        # update local config with real time parameters from server and dynamically replace variables value
        unzip_package_path, fedml_config_object = self.update_local_fedml_config(run_id, run_config)
        if unzip_package_path is None or fedml_config_object is None:
            self.cleanup_run_when_starting_failed()
            return

        entry_file_config = fedml_config_object["entry_config"]
        dynamic_args_config = fedml_config_object["dynamic_args"]
        entry_file = str(entry_file_config["entry_file"]).replace('\\', os.sep).replace('/', os.sep)
        entry_file = os.path.basename(entry_file)
        conf_file = entry_file_config["conf_file"]
        conf_file = str(conf_file).replace('\\', os.sep).replace('/', os.sep)
        ServerConstants.cleanup_learning_process()
        if not os.path.exists(unzip_package_path):
            self.cleanup_run_when_starting_failed()
            return
        os.chdir(os.path.join(unzip_package_path, "fedml"))

        time.sleep(3)

        python_program = get_python_program()
        process = ServerConstants.exec_console_with_shell_script_list(
            [
                python_program,
                entry_file,
                "--cf",
                conf_file,
                "--rank",
                str(dynamic_args_config["rank"]),
                "--role",
                "server",
            ],
            should_capture_stdout=False,
            should_capture_stderr=True
        )
        ServerConstants.save_learning_process(process.pid)
        self.release_client_mqtt_mgr()
        ret_code, out, err = ServerConstants.get_console_pipe_out_err_results(process)
        if ret_code is None or ret_code == 0:
            out_str = out.decode(encoding="utf-8")
            if out_str != "":
                logging.info("{}".format(out_str))

            sys_utils.log_return_info(entry_file, 0)
        else:
            if err is not None:
                err_str = err.decode(encoding="utf-8")
                if err_str != "":
                    logging.error("{}".format(err_str))

            sys_utils.log_return_info(entry_file, ret_code)

            self.stop_run_when_starting_failed()

    def reset_all_devices_status(self):
        edge_id_list = self.request_json["edgeids"]
        for edge_id in edge_id_list:
            self.mlops_metrics.report_client_training_status(edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE)

    def set_all_devices_status(self, status):
        edge_id_list = self.request_json["edgeids"]
        for edge_id in edge_id_list:
            self.mlops_metrics.broadcast_client_training_status(edge_id, status)

    def stop_run(self):
        self.setup_client_mqtt_mgr()

        edge_id_list = self.request_json["edgeids"]
        self.send_training_stop_request_to_edges(edge_id_list, json.dumps(self.request_json))

        logging.info("Stop run successfully.")

        time.sleep(4)

        self.mlops_metrics.report_server_training_status(self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_KILLED)

        time.sleep(1)

        ServerConstants.cleanup_learning_process()

        try:
            local_package_path = ServerConstants.get_package_download_dir()
            for package_file in listdir(local_package_path):
                if os.path.basename(package_file).startswith("run_" + str(self.run_id)):
                    shutil.rmtree(os.path.join(local_package_path, package_file), ignore_errors=True)
        except Exception as e:
            pass

        self.release_client_mqtt_mgr()

    def stop_run_when_starting_failed(self):
        self.setup_client_mqtt_mgr()

        edge_id_list = self.request_json["edgeids"]
        logging.info("edge ids {}".format(str(edge_id_list)))
        self.send_exit_train_with_exception_request_to_edges(edge_id_list, json.dumps(self.request_json))

        # logging.info("Stop run successfully when starting failed.")

        time.sleep(4)

        self.mlops_metrics.report_server_id_status(self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED)

        time.sleep(1)

        self.set_all_devices_status(ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED)
        self.mlops_metrics.server_send_stop_train_msg()

        self.release_client_mqtt_mgr()

    def cleanup_run_when_finished(self):
        if self.run_as_cloud_agent:
            self.stop_cloud_server()

        self.setup_client_mqtt_mgr()

        self.wait_client_mqtt_connected()

        logging.info("Cleanup run successfully when finished.")

        self.mlops_metrics.broadcast_server_training_status(
            self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED
        )

        try:
            self.mlops_metrics.set_sys_reporting_status(False)
        except Exception as ex:
            pass

        time.sleep(1)

        ServerConstants.cleanup_learning_process()

        try:
            local_package_path = ServerConstants.get_package_download_dir()
            for package_file in listdir(local_package_path):
                if os.path.basename(package_file).startswith("run_" + str(self.run_id)):
                    shutil.rmtree(os.path.join(local_package_path, package_file), ignore_errors=True)
        except Exception as e:
            pass

        self.release_client_mqtt_mgr()

    def cleanup_run_when_starting_failed(self):
        if self.run_as_cloud_agent:
            self.stop_cloud_server()

        self.setup_client_mqtt_mgr()

        self.wait_client_mqtt_connected()

        logging.info("Cleanup run successfully when starting failed.")

        self.mlops_metrics.broadcast_server_training_status(self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED)

        try:
            self.mlops_metrics.set_sys_reporting_status(False)
        except Exception as ex:
            pass

        time.sleep(1)

        ServerConstants.cleanup_learning_process()

        try:
            local_package_path = ServerConstants.get_package_download_dir()
            for package_file in listdir(local_package_path):
                if os.path.basename(package_file).startswith("run_" + str(self.run_id)):
                    shutil.rmtree(os.path.join(local_package_path, package_file), ignore_errors=True)
        except Exception as e:
            pass

        self.release_client_mqtt_mgr()

    def send_training_request_to_edges(self):
        self.wait_client_mqtt_connected()

        run_id = self.request_json["runId"]
        edge_id_list = self.request_json["edgeids"]
        logging.info("Edge ids: " + str(edge_id_list))
        for edge_id in edge_id_list:
            topic_start_train = "flserver_agent/" + str(edge_id) + "/start_train"
            logging.info("start_train: send topic " + topic_start_train + " to client...")
            self.client_mqtt_mgr.send_message(topic_start_train, json.dumps(self.request_json))

    def callback_client_status_msg(self, topic=None, payload=None):
        payload_json = json.loads(payload)
        run_id = payload_json["run_id"]
        edge_id = payload_json["edge_id"]
        status = payload_json["status"]
        edge_id_list = self.request_json["edgeids"]
        if len(edge_id_list) == 1 and edge_id_list[0] == edge_id and \
                status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED:
            if self.run_as_edge_server_and_agent or self.run_as_cloud_agent:
                server_runner = FedMLServerRunner(
                    self.args, run_id=run_id, request_json=self.request_json, agent_config=self.agent_config
                )
                server_runner.edge_id = self.edge_id
                server_runner.run_as_edge_server_and_agent = self.run_as_edge_server_and_agent
                server_runner.run_as_cloud_agent = self.run_as_cloud_agent
                server_runner.run_status = ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED
                Process(target=server_runner.cleanup_client_with_status).start()

    def callback_start_train(self, topic=None, payload=None):
        logging.info("callback_start_train from Web: {}".format(payload))

        # get training params
        if self.run_as_cloud_server:
            message_bytes = payload.encode("ascii")
            base64_bytes = base64.b64decode(message_bytes)
            payload = base64_bytes.decode("ascii")
            logging.info("decoded payload: {}".format(payload))
        request_json = json.loads(payload)
        self.start_request_json = payload
        run_id = request_json["runId"]
        self.run_id = run_id
        ServerConstants.save_runner_infos(self.args.device_id + "." + self.args.os_name, self.edge_id, run_id=run_id)

        # Start server with multi processing mode
        self.request_json = request_json
        self.running_request_json[str(run_id)] = request_json

        if not self.run_as_cloud_server:
            # Setup MQTT message listener to the client status message from the server.
            edge_id_list = request_json["edgeids"]
            for edge_id in edge_id_list:
                topic_name = "fl_client/flclient_agent_" + str(edge_id) + "/status"
                self.mqtt_mgr.add_message_listener(topic_name, self.callback_client_status_msg)
                self.mqtt_mgr.subscribe_msg(topic_name)

        if self.run_as_edge_server_and_agent:
            # Start log processor for current run
            MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(run_id, self.edge_id)
            self.args.run_id = run_id

            server_runner = FedMLServerRunner(
                self.args, run_id=run_id, request_json=request_json, agent_config=self.agent_config
            )
            server_runner.run_as_edge_server_and_agent = self.run_as_edge_server_and_agent
            server_runner.edge_id = self.edge_id
            server_runner.start_request_json = self.start_request_json
            server_process = Process(target=server_runner.run)
            server_process.start()
            ServerConstants.save_run_process(server_process.pid)
        elif self.run_as_cloud_agent:
            # Start log processor for current run
            MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(
                run_id, self.request_json.get("cloudServerDeviceId", "0")
            )

            server_runner = FedMLServerRunner(
                self.args, run_id=run_id, request_json=request_json, agent_config=self.agent_config
            )
            server_runner.run_as_cloud_agent = self.run_as_cloud_agent
            server_runner.start_request_json = self.start_request_json
            server_process = Process(target=server_runner.start_cloud_server_process)
            server_process.start()
            ServerConstants.save_run_process(server_process.pid)
        elif self.run_as_cloud_server:
            self.server_agent_id = self.request_json.get("cloud_agent_id", self.edge_id)
            self.start_request_json = json.dumps(self.request_json)
            run_id = self.request_json["runId"]

            # Init local database
            FedMLServerDataInterface.get_instance().create_job_table()

            # Start log processor for current run
            self.args.run_id = run_id
            MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(run_id, self.edge_id)
            self.run()

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
                + ";envsubst < "
                + os.path.join(cur_dir, "templates", "fedml-server-svc.yaml")
                + " | kubectl apply -f - "
        )
        logging.info("FedMLServerRunner.run with k8s: " + run_deployment_cmd)
        os.system(run_deployment_cmd)

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
        if self.client_mqtt_mgr is not None:
            self.client_mqtt_lock.acquire()
            self.client_mqtt_mgr.remove_disconnected_listener(self.on_client_mqtt_disconnected)
            self.client_mqtt_is_connected = False
            self.client_mqtt_mgr.disconnect()
            self.client_mqtt_mgr = None
            self.client_mqtt_lock.release()

        logging.info(
            "client agent config: {},{}".format(
                self.agent_config["mqtt_config"]["BROKER_HOST"], self.agent_config["mqtt_config"]["BROKER_PORT"]
            )
        )

        self.client_mqtt_mgr = MqttManager(
            self.agent_config["mqtt_config"]["BROKER_HOST"],
            self.agent_config["mqtt_config"]["BROKER_PORT"],
            self.agent_config["mqtt_config"]["MQTT_USER"],
            self.agent_config["mqtt_config"]["MQTT_PWD"],
            self.agent_config["mqtt_config"]["MQTT_KEEPALIVE"],
            "FedML_ServerAgent_Metrics_{}_{}".format(self.args.current_device_id, str(os.getpid()))
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

    def release_client_mqtt_mgr(self, real_release=False):
        if real_release:
            if self.client_mqtt_mgr is not None:
                self.client_mqtt_mgr.disconnect()
                self.client_mqtt_mgr.loop_stop()

            self.client_mqtt_lock.acquire()
            if self.client_mqtt_mgr is not None:
                self.client_mqtt_is_connected = False
                self.client_mqtt_mgr = None
            self.client_mqtt_lock.release()

    def wait_client_mqtt_connected(self):
        pass
        # while True:
        #     self.client_mqtt_lock.acquire()
        #     if self.client_mqtt_is_connected is True:
        #         self.mlops_metrics.set_messenger(self.client_mqtt_mgr)
        #         self.mlops_metrics.run_id = self.run_id
        #         self.mlops_metrics.edge_id = self.edge_id
        #         self.mlops_metrics.server_agent_id = self.server_agent_id
        #         self.client_mqtt_lock.release()
        #         break
        #     self.client_mqtt_lock.release()
        #     time.sleep(0.1)

    def send_training_stop_request_to_edges(self, edge_id_list, payload):
        self.wait_client_mqtt_connected()
        for edge_id in edge_id_list:
            topic_stop_train = "flserver_agent/" + str(edge_id) + "/stop_train"
            logging.info("stop_train: send topic " + topic_stop_train)
            self.client_mqtt_mgr.send_message(topic_stop_train, payload)
            self.mlops_metrics.server_broadcast_client_training_status(edge_id,
                                                                       ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED)

    def send_exit_train_with_exception_request_to_edges(self, edge_id_list, payload):
        self.wait_client_mqtt_connected()
        for edge_id in edge_id_list:
            topic_exit_train = "flserver_agent/" + str(edge_id) + "/exit_train_with_exception"
            logging.info("exit_train_with_exception: send topic " + topic_exit_train)
            self.client_mqtt_mgr.send_message(topic_exit_train, payload)
            self.mlops_metrics.server_broadcast_client_training_status(edge_id,
                                                                       ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED)

    def callback_stop_train(self, topic, payload):
        # logging.info("callback_stop_train: topic = %s, payload = %s" % (topic, payload))

        request_json = json.loads(payload)
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

        # Stop cross-silo server with multi processing mode
        stop_request_json = self.running_request_json.get(str(run_id), None)
        if stop_request_json is None:
            stop_request_json = request_json
        if self.run_as_edge_server_and_agent:
            server_runner = FedMLServerRunner(
                self.args, run_id=run_id, request_json=stop_request_json, agent_config=self.agent_config,
                edge_id=self.edge_id
            )
            server_runner.run_as_edge_server_and_agent = self.run_as_edge_server_and_agent
            Process(target=server_runner.stop_run).start()

            # Stop log processor for current run
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, self.edge_id)
        elif self.run_as_cloud_agent:
            server_runner = FedMLServerRunner(
                self.args, run_id=run_id, request_json=stop_request_json, agent_config=self.agent_config,
                edge_id=server_id
            )
            server_runner.run_as_cloud_agent = self.run_as_cloud_agent
            Process(target=server_runner.stop_cloud_server_process).start()

            # Stop log processor for current run
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, server_id)
        elif self.run_as_cloud_server:
            # Stop log processor for current run
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, self.edge_id)
            pass

        if self.running_request_json.get(str(run_id), None) is not None:
            self.running_request_json.pop(str(run_id))

    def callback_runner_id_status(self, topic, payload):
        # logging.info("callback_runner_id_status: topic = %s, payload = %s" % (topic, payload))

        request_json = json.loads(payload)
        run_id = request_json["run_id"]
        status = request_json["status"]
        edge_id = request_json["edge_id"]

        if (
                status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED
                or status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED
        ):
            # logging.info("Received training finished message.")

            # logging.info("Will end training server.")

            # Stop cross-silo server with multi processing mode
            stop_request_json = self.running_request_json.get(str(run_id), None)
            if stop_request_json is None:
                stop_request_json = request_json
            if self.run_as_edge_server_and_agent:
                server_runner = FedMLServerRunner(
                    self.args, run_id=run_id, request_json=stop_request_json, agent_config=self.agent_config
                )
                server_runner.edge_id = self.edge_id
                server_runner.run_as_edge_server_and_agent = self.run_as_edge_server_and_agent
                server_runner.run_status = status
                status_process = Process(target=server_runner.cleanup_client_with_status)
                status_process.start()
                status_process.join(10)

                # Stop log processor for current run
                MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, self.edge_id)
            elif self.run_as_cloud_agent:
                server_runner = FedMLServerRunner(
                    self.args, run_id=run_id, request_json=stop_request_json, agent_config=self.agent_config
                )
                server_runner.run_as_cloud_agent = self.run_as_cloud_agent
                server_runner.edge_id = edge_id
                server_runner.run_status = status
                status_process = Process(target=server_runner.cleanup_client_with_status)
                status_process.start()
                status_process.join(10)

                # Stop log processor for current run
                MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, edge_id)
            elif self.run_as_cloud_server:
                # Stop log processor for current run
                MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, self.edge_id)
                pass

    def cleanup_client_with_status(self):
        if self.run_status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED:
            self.cleanup_run_when_finished()
        elif self.run_status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED:
            self.cleanup_run_when_starting_failed()

    def report_client_status(self):
        self.send_agent_active_msg()

    def callback_report_current_status(self, topic, payload):
        request_json = json.loads(payload)
        if self.run_as_edge_server_and_agent:
            self.send_agent_active_msg()
        elif self.run_as_cloud_agent:
            self.send_agent_active_msg()
        elif self.run_as_cloud_server:
            pass

    def callback_client_agent_last_will_msg(self, topic, payload):
        msg = json.loads(payload)
        edge_id = msg.get("ID", None)
        status = msg.get("status", ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE)
        if edge_id is not None:
            self.client_agent_active_list[edge_id] = status
            MLOpsStatus.get_instance().set_client_agent_status(edge_id, status)

    def callback_client_agent_active_msg(self, topic, payload):
        msg = json.loads(payload)
        edge_id = msg.get("ID", None)
        status = msg.get("status", ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE)
        if edge_id is not None:
            self.client_agent_active_list[edge_id] = status

    def callback_server_last_will_msg(self, topic, payload):
        msg = json.loads(payload)
        server_id = msg.get("ID", None)
        status = msg.get("status", ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE)
        if server_id is not None and status == ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE:
            if self.server_active_list.get(server_id, None) is not None:
                self.server_active_list.pop(server_id)

    def callback_server_active_msg(self, topic, payload):
        msg = json.loads(payload)
        server_id = msg.get("ID", None)
        status = msg.get("status", ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE)
        if server_id is not None:
            self.server_active_list[server_id] = status

    @staticmethod
    def process_ota_upgrade_msg():
        os.system("pip install -U fedml")

    def callback_server_ota_msg(self, topic, payload):
        request_json = json.loads(payload)
        cmd = request_json["cmd"]

        if cmd == ServerConstants.FEDML_OTA_CMD_UPGRADE:
            try:
                Process(target=FedMLServerRunner.process_ota_upgrade_msg).start()
            except Exception as e:
                pass
        elif cmd == ServerConstants.FEDML_OTA_CMD_RESTART:
            raise Exception("Restart runner...")

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

                def GetUUID():
                    guid = ""
                    try:
                        cmd = "wmic csproduct get uuid"
                        guid = str(subprocess.check_output(cmd))
                        pos1 = guid.find("\\n") + 2
                        guid = guid[pos1:-15]
                    except Exception as ex:
                        pass
                    return str(guid)

                device_id = str(GetUUID())
            elif "posix" in os.name:
                device_id = hex(uuid.getnode())
            else:
                device_id = subprocess.Popen(
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

    def bind_account_and_device_id(self, url, account_id, device_id, os_name):
        role = "edge_server"
        if self.run_as_edge_server_and_agent:
            role = "edge_server"
        elif self.run_as_cloud_agent:
            role = "cloud_agent"
        elif self.run_as_cloud_server:
            role = "cloud_server"

        ip = requests.get('https://checkip.amazonaws.com').text.strip()
        fedml_ver, exec_path, os_ver, cpu_info, python_ver, torch_ver, mpi_installed, \
        cpu_usage, available_mem, total_mem, gpu_info, gpu_available_mem, gpu_total_mem = get_sys_runner_info()
        json_params = {
            "accountid": account_id,
            "deviceid": device_id,
            "type": os_name,
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
        status_code = response.json().get("code")
        if status_code == "SUCCESS":
            edge_id = response.json().get("data").get("id")
        else:
            return 0
        return edge_id

    def fetch_configs(self):
        return MLOpsConfigs.get_instance(self.args).fetch_all_configs()

    def send_agent_active_msg(self):
        active_topic = "/flserver_agent/active"
        status = MLOpsStatus.get_instance().get_server_agent_status(self.edge_id)
        if (
                status is not None
                and status != ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE
                and status != ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE
        ):
            return
        status = ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE
        active_msg = {"ID": self.edge_id, "status": status}
        MLOpsStatus.get_instance().set_server_agent_status(self.edge_id, status)
        self.mqtt_mgr.send_message_json(active_topic, json.dumps(active_msg))

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
        topic_report_status = "/mlops/report_device_status"
        self.mqtt_mgr.add_message_listener(topic_report_status, self.callback_report_current_status)

        # Setup MQTT message listener to the last will message from the client agent.
        topic_client_agent_last_will_msg = "/flclient_agent/last_will_msg"
        self.mqtt_mgr.add_message_listener(topic_client_agent_last_will_msg, self.callback_client_agent_last_will_msg)

        # Setup MQTT message listener to the active status message from the client agent.
        topic_client_agent_active_msg = "/flclient_agent/active"
        self.mqtt_mgr.add_message_listener(topic_client_agent_active_msg, self.callback_client_agent_active_msg)

        # Setup MQTT message listener to the last will message from the server.
        topic_server_last_will_msg = "/flserver/last_will_msg"
        self.mqtt_mgr.add_message_listener(topic_server_last_will_msg, self.callback_server_last_will_msg)

        # Setup MQTT message listener to the active status message from the server.
        topic_server_active_msg = "/flserver/active"
        self.mqtt_mgr.add_message_listener(topic_server_active_msg, self.callback_server_active_msg)

        # Setup MQTT message listener to OTA messages from the MLOps.
        topic_ota_msg = "/mlops/flserver_agent_" + str(server_agent_id) + "/ota"
        self.mqtt_mgr.add_message_listener(topic_ota_msg, self.callback_server_ota_msg)

        # Subscribe topics for starting train, stopping train and fetching client status.
        mqtt_client_object.subscribe(topic_start_train, qos=2)
        mqtt_client_object.subscribe(topic_stop_train, qos=2)
        mqtt_client_object.subscribe(topic_server_status, qos=2)
        mqtt_client_object.subscribe(topic_report_status, qos=2)
        mqtt_client_object.subscribe(topic_client_agent_last_will_msg, qos=2)
        mqtt_client_object.subscribe(topic_client_agent_active_msg, qos=2)
        mqtt_client_object.subscribe(topic_server_last_will_msg, qos=2)
        mqtt_client_object.subscribe(topic_server_active_msg, qos=2)
        mqtt_client_object.subscribe(topic_ota_msg, qos=2)

        # Broadcast the first active message.
        # self.send_agent_active_msg()

        # Echo results
        click.echo("")
        click.echo("Congratulations, you have logged into the FedML MLOps platform successfully!")
        click.echo("Your server unique device id is " + str(self.unique_device_id))

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
            "/flserver_agent/last_will_msg",
            json.dumps({"ID": self.edge_id, "status": ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE}),
        )

        # Init local database
        FedMLServerDataInterface.get_instance().create_job_table()

        # Start local API services
        local_api_process = ServerConstants.exec_console_with_script(
            "uvicorn fedml.cli.server_deployment.server_api:api --host 0.0.0.0 --port {} --reload".format(
                ServerConstants.LOCAL_SERVER_API_PORT),
            should_capture_stdout=False,
            should_capture_stderr=False
        )

        # Setup MQTT connected listener
        self.mqtt_mgr.add_connected_listener(self.on_agent_mqtt_connected)
        self.mqtt_mgr.add_disconnected_listener(self.on_agent_mqtt_disconnected)
        self.mqtt_mgr.connect()

        self.setup_client_mqtt_mgr()
        self.wait_client_mqtt_connected()
        self.mlops_metrics.report_server_training_status(self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE)
        MLOpsStatus.get_instance().set_server_agent_status(
            self.edge_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE
        )
        self.release_client_mqtt_mgr()

        MLOpsRuntimeLogDaemon.get_instance(self.args).stop_all_log_processor()

    def start_agent_mqtt_loop(self):
        # Start MQTT message loop
        try:
            self.mqtt_mgr.loop_forever()
        except Exception as e:
            pass
