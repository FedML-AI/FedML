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
from urllib.parse import urlparse

import requests
from fedml import mlops
from fedml.cli.model_deployment.device_model_msg_object import FedMLModelMsgObject
from fedml.core.distributed.communication.s3.remote_storage import S3Storage
from ..comm_utils import sys_utils

from ...core.mlops.mlops_runtime_log import MLOpsRuntimeLog

from ...core.distributed.communication.mqtt.mqtt_manager import MqttManager
from ...cli.comm_utils.yaml_utils import load_yaml_config
from ...cli.model_deployment.device_client_constants import ClientConstants

from ...core.mlops.mlops_metrics import MLOpsMetrics

from ...core.mlops.mlops_configs import MLOpsConfigs
from ...core.mlops.mlops_runtime_log_daemon import MLOpsRuntimeLogDaemon
from ...core.mlops.mlops_status import MLOpsStatus
from ..comm_utils.sys_utils import get_sys_runner_info, get_python_program
from .device_model_deployment import start_deployment
from .device_client_data_interface import FedMLClientDataInterface
from ...serving.fedml_client import FedMLModelServingClient
from ...core.mlops.mlops_utils import MLOpsUtils


class RunnerError(Exception):
    """ Runner failed. """
    pass


class FedMLClientRunner:
    FEDML_BOOTSTRAP_RUN_OK = "[FedML]Bootstrap Finished"

    def __init__(self, args, edge_id=0, request_json=None, agent_config=None, run_id=0):
        self.run_process_event = None
        self.device_status = None
        self.current_training_status = None
        self.mqtt_mgr = None
        self.client_mqtt_mgr = None
        self.client_mqtt_is_connected = False
        self.client_mqtt_lock = None
        self.edge_id = edge_id
        self.run_id = run_id
        self.unique_device_id = None
        self.process = None
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

        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES = {}

        self.mlops_metrics = None
        self.client_active_list = dict()
        self.infer_host = "127.0.0.1"
        self.model_is_from_open = False

        self.model_runner_mapping = dict()
        self.ntp_offset = MLOpsUtils.get_ntp_offset()

    def unzip_file(self, zip_file, unzip_file_path) -> str:
        unziped_file_name = ""
        if zipfile.is_zipfile(zip_file):
            with zipfile.ZipFile(zip_file, "r") as zipf:
                zipf.extractall(unzip_file_path)
                unziped_file_name = zipf.namelist()[0]

        return unziped_file_name

    def retrieve_and_unzip_package(self, package_name, package_url):
        local_package_path = ClientConstants.get_model_package_dir()
        if not os.path.exists(local_package_path):
            os.makedirs(local_package_path, exist_ok=True)
        local_package_file = "{}.zip".format(os.path.join(local_package_path, package_name))
        if os.path.exists(local_package_file):
            os.remove(local_package_file)
        urllib.request.urlretrieve(package_url, filename=None, reporthook=self.package_download_progress)   # Do NOT rename, use the filename from MLOps
        unzip_package_path = ClientConstants.get_model_dir()
        self.fedml_packages_base_dir = unzip_package_path
        try:
            shutil.rmtree(
                os.path.join(unzip_package_path, package_name), ignore_errors=True
            )
        except Exception as e:
            pass
        logging.info("local_package_file {}, unzip_package_path {}".format(
            local_package_file, unzip_package_path))
        package_name = self.unzip_file(local_package_file, unzip_package_path)  # Using unziped folder name
        unzip_package_path = os.path.join(unzip_package_path, package_name)
        model_bin_file = os.path.join(unzip_package_path, "fedml_model.bin")
        return unzip_package_path, model_bin_file

    def retrieve_binary_model_file(self, package_name, package_url):
        local_package_path = ClientConstants.get_model_package_dir()
        if not os.path.exists(local_package_path):
            os.makedirs(local_package_path, exist_ok=True)
        unzip_package_path = ClientConstants.get_model_dir()
        local_package_file = "{}".format(os.path.join(local_package_path, package_name))
        if os.path.exists(local_package_file):
            os.remove(local_package_file)
        urllib.request.urlretrieve(package_url, local_package_file, reporthook=self.package_download_progress)

        unzip_package_path = os.path.join(unzip_package_path, package_name)
        if not os.path.exists(unzip_package_path):
            os.makedirs(unzip_package_path, exist_ok=True)
        dst_model_file = os.path.join(unzip_package_path, package_name)
        if os.path.exists(local_package_file):
            shutil.copy(local_package_file, dst_model_file)

        return unzip_package_path, dst_model_file

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

    def build_dynamic_constrain_variables(self, run_id, run_config):
        pass

    def update_local_fedml_config(self, run_id, model_config, model_config_parameters):
        model_name = model_config["model_name"]
        model_storage_url = model_config["model_storage_url"]
        scale_min = model_config["instance_scale_min"]
        scale_max = model_config["instance_scale_max"]
        inference_engine = model_config.get("inference_engine", 0)
        inference_end_point_id = run_id

        # Retrieve model package or model binary file.
        if self.model_is_from_open:
            unzip_package_path, model_bin_file = self.retrieve_binary_model_file(model_name, model_storage_url)
        else:
            unzip_package_path, model_bin_file = self.retrieve_and_unzip_package(model_name, model_storage_url)

        # Load the config to memory
        package_conf_object = {}
        fedml_local_config_file = os.path.join(unzip_package_path, "fedml_model_config.yaml")
        if model_config_parameters is not None:
            package_conf_object = model_config_parameters
            with open(fedml_local_config_file, 'w') as f:
                json.dump(package_conf_object, f)
        else:
            if os.path.exists(fedml_local_config_file):
                package_conf_object = load_yaml_config(fedml_local_config_file)

        return unzip_package_path, model_bin_file, package_conf_object

    def build_dynamic_args(self, run_config, package_conf_object, base_dir):
        pass

    def download_model_package(self, package_name, package_url):
        # Copy config file from the client
        unzip_package_path = self.retrieve_and_unzip_package(
            package_name, package_url
        )

        return unzip_package_path

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
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(self.run_id, self.edge_id)
            if self.mlops_metrics is not None:
                self.mlops_metrics.stop_sys_perf()
            time.sleep(3)
            sys_utils.cleanup_all_fedml_client_login_processes(ClientConstants.CLIENT_LOGIN_PROGRAM,
                                                               clean_process_group=False)
            sys.exit(1)
        finally:
            logging.info("Release resources.")
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(self.run_id, self.edge_id)
            if self.mlops_metrics is not None:
                self.mlops_metrics.stop_sys_perf()
            time.sleep(3)
            self.release_client_mqtt_mgr()

    def check_runner_stop_event(self):
        if self.run_process_event.is_set():
            logging.info("Received stopping event.")
            raise RunnerError("Runner stopped")

    def inference_run(self):
        run_id, end_point_name, token, user_id, user_name, device_ids, device_objs, model_config, model_name, \
            model_id, model_storage_url, scale_min, scale_max, inference_engine, model_is_from_open, \
            inference_end_point_id, use_gpu, memory_size, model_version = self.parse_model_run_params(self.request_json)

        inference_client = FedMLModelServingClient(self.args,
                                                   end_point_name,
                                                   model_name,
                                                   model_version,
                                                   inference_request=self.request_json)
        inference_client.run()

    def run_impl(self):
        run_id = self.request_json["end_point_id"]
        end_point_name = self.request_json["end_point_name"]
        token = self.request_json["token"]
        user_id = self.request_json["user_id"]
        user_name = self.request_json["user_name"]
        device_ids = self.request_json["device_ids"]
        device_objs = self.request_json["device_objs"]

        model_config = self.request_json["model_config"]
        model_name = model_config["model_name"]
        model_id = model_config["model_id"]
        model_version = model_config["model_version"]
        model_storage_url = model_config["model_storage_url"]
        scale_min = model_config["instance_scale_min"]
        scale_max = model_config["instance_scale_max"]
        inference_engine = model_config.get("inference_engine", ClientConstants.INFERENCE_ENGINE_TYPE_INT_TRITON)
        self.model_is_from_open = True if model_config.get("is_from_open", 0) == 1 else False
        if self.model_is_from_open:
            model_net_url = model_config["model_net_url"]
        model_config_parameters = self.request_json["parameters"]
        inference_end_point_id = run_id
        use_gpu = "gpu"  # TODO: Get GPU from device infos
        memory_size = "4096m"  # TODO: Get Memory size for each instance

        self.check_runner_stop_event()

        logging.info("model deployment request: {}".format(self.request_json))

        MLOpsRuntimeLog.get_instance(self.args).init_logs(show_stdout_log=True)

        # Initiate an FedMLInferenceClient object
        # client_runner = FedMLClientRunner(
        #     self.args, edge_id=self.edge_id, run_id=self.run_id, request_json=self.request_json,
        #     agent_config=self.agent_config
        # )
        # inference_process = Process(target=client_runner.inference_run)
        # inference_process.start()

        self.mlops_metrics.report_client_training_status(self.edge_id,
                                                         ClientConstants.MSG_MLOPS_CLIENT_STATUS_INITIALIZING,
                                                         is_from_model=True,
                                                         running_json=json.dumps(self.request_json))

        self.mlops_metrics.report_client_training_status(self.edge_id,
                                                         ClientConstants.MSG_MLOPS_CLIENT_STATUS_RUNNING,
                                                         is_from_model=True)

        self.check_runner_stop_event()

        # update local config with real time parameters from server and dynamically replace variables value
        logging.info("download and unzip model to local...")
        unzip_package_path, model_bin_file, fedml_config_object = \
            self.update_local_fedml_config(run_id, model_config, model_config_parameters)
        if unzip_package_path is None or fedml_config_object is None:
            logging.info("failed to update local fedml config.")
            self.check_runner_stop_event()
            self.cleanup_run_when_starting_failed()
            self.mlops_metrics.client_send_exit_train_msg(run_id, self.edge_id,
                                                          ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED)
            return

        logging.info("check downloaded packages...")
        if not os.path.exists(unzip_package_path):
            logging.info("failed to unzip file.")
            self.check_runner_stop_event()
            self.cleanup_run_when_starting_failed()
            self.mlops_metrics.client_send_exit_train_msg(run_id, self.edge_id,
                                                          ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED)
            return

        # download model net and load into the torch model
        model_from_open = None
        if self.model_is_from_open:
            logging.info("process the model net from open...")
            self.check_runner_stop_event()
            s3_config = self.agent_config.get("s3_config", None)
            if s3_config is not None and model_net_url is not None and model_net_url != "":
                s3_client = S3Storage(s3_config)
                url_parsed = urlparse(model_net_url)
                path_list = url_parsed.path.split("/")
                if len(path_list) > 0:
                    model_key = path_list[-1]
                    model_from_open = s3_client.read_model_net(model_key,
                                                               ClientConstants.get_model_cache_dir())

                model_input_size, model_input_type = mlops.get_training_model_input_info(model_net_url, s3_config)
                if model_input_size is not None and model_input_type is not None:
                    model_config_parameters["input_size"] = model_input_size
                    model_config_parameters["input_types"] = model_input_type
                    logging.info(
                        f"model input size {model_input_size}, input type {model_input_type} from the open platform.")

        logging.info("start the model deployment...")
        self.check_runner_stop_event()
        running_model_name, inference_output_url, inference_model_version, model_metadata, model_config = \
            start_deployment(
                inference_end_point_id, end_point_name, model_id, model_version,
                unzip_package_path, model_bin_file, model_name, inference_engine,
                ClientConstants.INFERENCE_HTTP_PORT,
                ClientConstants.INFERENCE_GRPC_PORT,
                ClientConstants.INFERENCE_METRIC_PORT,
                use_gpu, memory_size,
                ClientConstants.INFERENCE_CONVERTOR_IMAGE,
                ClientConstants.INFERENCE_SERVER_IMAGE,
                self.infer_host,
                self.model_is_from_open, model_config_parameters,
                model_from_open,
                token)
        if inference_output_url == "":
            logging.error("failed to deploy the model...")
            self.send_deployment_status(end_point_name, self.edge_id,
                                        model_id, model_name, model_version,
                                        inference_output_url,
                                        ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED)
            self.send_deployment_results(end_point_name, self.edge_id,
                                         ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED,
                                         model_id, model_name, inference_output_url,
                                         inference_model_version, ClientConstants.INFERENCE_HTTP_PORT,
                                         inference_engine, model_metadata, model_config)
            self.mlops_metrics.run_id = self.run_id
            self.mlops_metrics.broadcast_client_training_status(self.edge_id,
                                                                ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED,
                                                                is_from_model=True)

            self.mlops_metrics.client_send_exit_train_msg(run_id, self.edge_id,
                                                          ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED)
        else:
            logging.info("finished deployment, continue to send results to master...")
            self.send_deployment_status(end_point_name, self.edge_id,
                                        model_id, model_name, model_version,
                                        inference_output_url,
                                        ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED)
            self.send_deployment_results(end_point_name, self.edge_id,
                                         ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED,
                                         model_id, model_name, inference_output_url,
                                         model_version, ClientConstants.INFERENCE_HTTP_PORT,
                                         inference_engine, model_metadata, model_config)
            time.sleep(1)
            self.mlops_metrics.run_id = self.run_id
            self.mlops_metrics.broadcast_client_training_status(self.edge_id,
                                                                ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED,
                                                                is_from_model=True)

    def send_deployment_results(self, end_point_name, device_id, model_status,
                                model_id, model_name, model_inference_url,
                                model_version, inference_port, inference_engine,
                                model_metadata, model_config):
        deployment_results_topic = "model_ops/model_device/return_deployment_result/{}".format(device_id)
        deployment_results_payload = {"end_point_id": self.run_id, "end_point_name": end_point_name,
                                      "model_id": model_id, "model_name": model_name,
                                      "model_url": model_inference_url, "model_version": model_version,
                                      "port": inference_port,
                                      "inference_engine": inference_engine,
                                      "model_metadata": model_metadata,
                                      "model_config": model_config,
                                      "model_status": model_status}

        logging.info("send_deployment_results: topic {}, payload {}.".format(deployment_results_topic,
                                                                             deployment_results_payload))
        self.client_mqtt_mgr.send_message_json(deployment_results_topic, json.dumps(deployment_results_payload))

    def send_deployment_status(self, end_point_name, device_id,
                               model_id, model_name, model_version,
                               model_inference_url, model_status):
        deployment_status_topic = "model_ops/model_device/return_deployment_status/{}".format(device_id)
        deployment_status_payload = {"end_point_id": self.run_id, "end_point_name": end_point_name,
                                     "device_id": device_id,
                                     "model_id": model_id, "model_name": model_name,
                                     "model_version": model_version,
                                     "model_url": model_inference_url, "model_status": model_status}

        logging.info("send_deployment_status: topic {}, payload {}.".format(deployment_status_topic,
                                                                            deployment_status_payload))
        self.client_mqtt_mgr.send_message_json(deployment_status_topic, json.dumps(deployment_status_payload))

    def reset_devices_status(self, edge_id, status):
        self.mlops_metrics.run_id = self.run_id
        self.mlops_metrics.edge_id = edge_id
        self.mlops_metrics.broadcast_client_training_status(edge_id, status, is_from_model=True)

    def cleanup_run_when_starting_failed(self):
        logging.info("Cleanup run successfully when starting failed.")

        self.reset_devices_status(self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED)

        time.sleep(2)

        try:
            self.mlops_metrics.stop_sys_perf()
        except Exception as ex:
            pass

        time.sleep(1)

    def cleanup_run_when_finished(self):
        logging.info("Cleanup run successfully when finished.")

        self.reset_devices_status(self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED)

        time.sleep(2)

        try:
            self.mlops_metrics.stop_sys_perf()
        except Exception as ex:
            pass

        time.sleep(1)

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
            "FedML_ModelClientAgent_Metrics_{}_{}_{}".format(self.args.current_device_id,
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
        run_id = request_json["end_point_id"]

        try:
            parameters = request_json.get("parameters", None)
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

            logging.info(f"Upgrade to version {upgrade_version} ...")

            sys_utils.do_upgrade(self.version, upgrade_version)

            raise Exception("Restarting after upgraded...")

    def callback_start_deployment(self, topic, payload):
        """
        topic: model_ops/model_device/start_deployment/model-agent-device-id
        payload: {"model_name": "image-model", "model_storage_url":"s3-url", "instance_scale_min":1, "instance_scale_max":3, "inference_engine":"onnx (or tensorrt)"}
        """
        # get deployment params
        request_json = json.loads(payload)
        run_id = request_json["end_point_id"]
        token = request_json["token"]
        user_id = request_json["user_id"]
        user_name = request_json["user_name"]
        device_ids = request_json["device_ids"]
        device_objs = request_json["device_objs"]

        model_config = request_json["model_config"]
        model_name = model_config["model_name"]
        model_storage_url = model_config["model_storage_url"]
        scale_min = model_config["instance_scale_min"]
        scale_max = model_config["instance_scale_max"]
        inference_engine = model_config.get("inference_engine", 0)
        inference_end_point_id = run_id

        try:
            _, _ = MLOpsConfigs.get_instance(self.args).fetch_configs()
        except Exception as e:
            pass

        # Terminate previous process about starting or stopping run command
        ClientConstants.cleanup_run_process(run_id)
        ClientConstants.save_runner_infos(self.args.device_id + "." + self.args.os_name, self.edge_id, run_id=run_id)

        # Start log processor for current run
        run_id = inference_end_point_id
        self.args.run_id = run_id
        self.args.edge_id = self.edge_id
        MLOpsRuntimeLog.get_instance(self.args).init_logs(show_stdout_log=True)
        MLOpsRuntimeLogDaemon.get_instance(self.args).set_log_source(
            ClientConstants.FEDML_LOG_SOURCE_TYPE_MODEL_END_POINT)
        MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(run_id, self.edge_id)

        self.ota_upgrade(payload, request_json)

        # Start client with multiprocessing mode
        request_json["run_id"] = run_id
        self.request_json = request_json
        client_runner = FedMLClientRunner(
            self.args, edge_id=self.edge_id, request_json=request_json, agent_config=self.agent_config, run_id=run_id
        )
        client_runner.infer_host = self.infer_host
        if self.run_process_event is None:
            self.run_process_event = multiprocessing.Event()
        self.run_process_event.clear()
        client_runner.run_process_event = self.run_process_event
        self.model_runner_mapping[run_id] = client_runner
        self.run_id = run_id
        self.process = Process(target=client_runner.run, args=(self.run_process_event,))
        # client_runner.run()
        self.process.start()
        ClientConstants.save_run_process(run_id, self.process.pid)

    def set_runner_stopped_event(self, run_id):
        client_runner = self.model_runner_mapping.get(run_id, None)
        if client_runner is not None:
            if client_runner.run_process_event is not None:
                client_runner.run_process_event.set()
            self.model_runner_mapping.pop(run_id)

    def callback_delete_deployment(self, topic, payload):
        logging.info("callback_delete_deployment: topic = %s, payload = %s" % (topic, payload))

        # Parse payload as the model message object.
        model_msg_object = FedMLModelMsgObject(topic, payload)

        ClientConstants.remove_deployment(model_msg_object.end_point_name,
                                          model_msg_object.model_name, model_msg_object.model_version)

        self.set_runner_stopped_event(model_msg_object.run_id)

    def exit_run_with_exception_entry(self):
        try:
            self.setup_client_mqtt_mgr()
            self.exit_run_with_exception()
        except Exception as e:
            self.release_client_mqtt_mgr()
            sys_utils.cleanup_all_fedml_client_login_processes(
                ClientConstants.CLIENT_LOGIN_PROGRAM, clean_process_group=False)
            sys.exit(1)
        finally:
            self.release_client_mqtt_mgr()

    def exit_run_with_exception(self):
        logging.info("Exit run successfully.")

        ClientConstants.cleanup_learning_process(self.run_id)
        ClientConstants.cleanup_run_process(self.run_id)

        self.mlops_metrics.report_client_id_status(self.run_id, self.edge_id,
                                                   ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED,
                                                   is_from_model=True)

        time.sleep(1)

    def callback_exit_train_with_exception(self, topic, payload):
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
            self.cleanup_run_when_finished()
        elif self.device_status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED:
            self.cleanup_run_when_starting_failed()

    def callback_runner_id_status(self, topic, payload):
        logging.info("callback_runner_id_status: topic = %s, payload = %s" % (topic, payload))

        request_json = json.loads(payload)
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
            status_process = Process(target=client_runner.cleanup_client_with_status)
            status_process.start()
            status_process.join(15)

            # Stop log processor for current run
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, edge_id)

    def callback_report_current_status(self, topic, payload):
        self.send_agent_active_msg()

    @staticmethod
    def process_ota_upgrade_msg():
        os.system("pip install -U fedml")

    def callback_client_ota_msg(self, topic, payload):
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

    def bind_account_and_device_id(self, url, account_id, device_id, os_name, role="md.on_premise_device"):
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

    def recover_start_deployment_msg_after_upgrading(self):
        try:
            current_job = FedMLClientDataInterface.get_instance().get_current_job()
            if current_job is not None and \
                    current_job.status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING:
                logging.info("start deployment after upgrading.")
                topic_start_deployment = "model_ops/model_device/start_deployment/{}".format(str(self.edge_id))
                self.callback_start_deployment(topic_start_deployment, current_job.running_json)
        except Exception as e:
            logging.info("recover starting deployment message after upgrading: {}".format(traceback.format_exc()))

    def on_agent_mqtt_connected(self, mqtt_client_object):
        # The MQTT message topic format is as follows: <sender>/<receiver>/<action>

        # Setup MQTT message listener for starting deployment
        topic_start_deployment = "model_ops/model_device/start_deployment/{}".format(str(self.edge_id))
        self.mqtt_mgr.add_message_listener(topic_start_deployment, self.callback_start_deployment)

        # Setup MQTT message listener for delete deployment
        topic_delete_deployment = "model_ops/model_device/delete_deployment/{}".format(str(self.edge_id))
        self.mqtt_mgr.add_message_listener(topic_delete_deployment, self.callback_delete_deployment)

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

        # Subscribe topics for starting deployment, stopping deployment and fetching client status.
        mqtt_client_object.subscribe(topic_start_deployment, qos=2)
        mqtt_client_object.subscribe(topic_delete_deployment, qos=2)
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

        MLOpsRuntimeLog.get_instance(self.args).init_logs(show_stdout_log=True)

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
            "FedML_ModelClientAgent_Daemon_" + self.args.current_device_id,
            "flclient_agent/last_will_msg",
            json.dumps({"ID": self.edge_id, "status": ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE}),
        )
        self.agent_config = service_config

        # Init local database
        FedMLClientDataInterface.get_instance().create_job_table()

        # Start local API services
        python_program = get_python_program()
        local_api_process = ClientConstants.exec_console_with_script(
            "{} -m uvicorn fedml.cli.model_deployment.client_api:api --host 0.0.0.0 --port {} "
            "--log-level critical".format(python_program,
                                          ClientConstants.LOCAL_CLIENT_API_PORT),
            should_capture_stdout=False,
            should_capture_stderr=False
        )

        MLOpsRuntimeLogDaemon.get_instance(self.args).stop_all_log_processor()

        # Setup MQTT connected listener
        self.mqtt_mgr.add_connected_listener(self.on_agent_mqtt_connected)
        self.mqtt_mgr.add_disconnected_listener(self.on_agent_mqtt_disconnected)
        self.mqtt_mgr.connect()

        self.setup_client_mqtt_mgr()
        self.mlops_metrics.report_client_training_status(self.edge_id,
                                                         ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE,
                                                         is_from_model=True)
        MLOpsStatus.get_instance().set_client_agent_status(self.edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE)
        self.mlops_metrics.stop_sys_perf()
        setattr(self.args, "mqtt_config_path", service_config["mqtt_config"])
        self.mlops_metrics.report_sys_perf(self.args)

        self.recover_start_deployment_msg_after_upgrading()

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
