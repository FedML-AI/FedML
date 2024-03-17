import copy
import json
import logging
import multiprocessing
import platform
import sys

from multiprocessing import Process
import os
import shutil
import subprocess
import threading

import time
import traceback
import urllib
import uuid
import zipfile
from os import listdir

import requests
import torch

import fedml
from fedml.computing.scheduler.comm_utils.run_process_utils import RunProcessUtils

from ..comm_utils import sys_utils
from .device_server_data_interface import FedMLServerDataInterface
from ..scheduler_core.endpoint_sync_protocol import FedMLEndpointSyncProtocol
from ....core.mlops.mlops_runtime_log import MLOpsRuntimeLog

from ....core.distributed.communication.mqtt.mqtt_manager import MqttManager
from ..comm_utils.yaml_utils import load_yaml_config
from .device_client_constants import ClientConstants
from .device_server_constants import ServerConstants

from ....core.mlops.mlops_metrics import MLOpsMetrics

from ....core.mlops.mlops_configs import MLOpsConfigs
from ....core.mlops.mlops_runtime_log_daemon import MLOpsRuntimeLogDaemon
from ....core.mlops.mlops_status import MLOpsStatus
from ..comm_utils.sys_utils import get_sys_runner_info, get_python_program
from .device_model_cache import FedMLModelCache
from .device_model_msg_object import FedMLModelMsgObject
from ....core.mlops.mlops_utils import MLOpsUtils
from ..comm_utils.constants import SchedulerConstants
from .device_model_db import FedMLModelDatabase
from .device_replica_controller import FedMLDeviceReplicaController


class RunnerError(BaseException):
    """ Runner failed. """
    pass


class RunnerCompletedError(Exception):
    """ Runner completed. """
    pass


class FedMLServerRunner:
    FEDML_CLOUD_SERVER_PREFIX = "fedml-server-run-"

    def __init__(self, args, run_id=0, request_json=None, agent_config=None, edge_id=0):
        self.inference_gateway_process = None
        self.local_api_process = None
        self.run_process_event = None
        self.run_process_event_map = dict()
        self.run_process_completed_event = None
        self.run_process_completed_event_map = dict()
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

        self.agent_config = agent_config
        self.fedml_data_base_package_dir = os.path.join("/", "fedml", "data")
        self.fedml_data_local_package_dir = os.path.join("/", "fedml", "fedml-package", "fedml", "data")
        self.fedml_data_dir = self.fedml_data_base_package_dir
        self.fedml_config_dir = os.path.join("/", "fedml", "conf")

        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES = {}

        self.mlops_metrics = None
        self.run_status = None
        self.infer_host = "127.0.0.1"
        self.redis_addr = "local"
        self.redis_port = "6379"
        self.redis_password = "fedml_default"

        self.slave_deployment_statuses_mapping = dict()
        self.slave_deployment_results_mapping = dict()
        self.slave_update_result_mapping = dict()

        self.model_runner_mapping = dict()
        self.ntp_offset = MLOpsUtils.get_ntp_offset()

        self.subscribed_topics = list()
        self.user_name = None

        self.replica_controller = None
        self.deployed_replica_payload = None

    def build_dynamic_constrain_variables(self, run_id, run_config):
        pass

    def unzip_file(self, zip_file, unzip_file_path):
        unziped_file_name = ""
        if zipfile.is_zipfile(zip_file):
            with zipfile.ZipFile(zip_file, "r") as zipf:
                zipf.extractall(unzip_file_path)
                unziped_file_name = zipf.namelist()[0]

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
        local_package_path = ServerConstants.get_model_package_dir()
        if not os.path.exists(local_package_path):
            os.makedirs(local_package_path, exist_ok=True)
        local_package_file = "{}.zip".format(os.path.join(local_package_path, package_name))
        if os.path.exists(local_package_file):
            os.remove(local_package_file)

        # Download without renaming
        urllib.request.urlretrieve(package_url, filename=None, reporthook=self.package_download_progress)

        unzip_package_path = ServerConstants.get_model_dir()
        self.fedml_packages_base_dir = unzip_package_path
        try:
            shutil.rmtree(
                os.path.join(unzip_package_path, package_name), ignore_errors=True
            )
        except Exception as e:
            pass
        logging.info("local_package_file {}, unzip_package_path {}".format(
            local_package_file, unzip_package_path))
        package_name = self.unzip_file(local_package_file, unzip_package_path)
        unzip_package_path = os.path.join(unzip_package_path, package_name)
        return unzip_package_path

    def update_local_fedml_config(self, run_id, run_config):
        model_config = run_config
        model_name = model_config["model_name"]
        model_storage_url = model_config["model_storage_url"]
        scale_min = model_config.get("instance_scale_min", 0)
        scale_max = model_config.get("instance_scale_max", 0)
        inference_engine = model_config.get("inference_engine", 0)
        inference_end_point_id = run_id

        # Copy config file from the client
        unzip_package_path = self.retrieve_and_unzip_package(
            model_name, model_storage_url
        )
        fedml_local_config_file = os.path.join(unzip_package_path, "fedml_model_config.yaml")

        # Load the above config to memory
        package_conf_object = {}
        if os.path.exists(fedml_local_config_file):
            package_conf_object = load_yaml_config(fedml_local_config_file)

        return unzip_package_path, package_conf_object

    def get_usr_indicated_token(self, request_json) -> str:
        usr_indicated_token = ""
        if "parameters" in request_json and "authentication_token" in request_json["parameters"]:
            usr_indicated_token = request_json["parameters"]["authentication_token"]
        return usr_indicated_token

    def build_dynamic_args(self, run_config, package_conf_object, base_dir):
        pass

    def run(self, process_event, completed_event):
        # print(f"Model master runner process id {os.getpid()}, run id {self.run_id}")

        if platform.system() != "Windows":
            os.setsid()

        os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
        os.environ.setdefault('PYTHONWARNINGS', 'ignore:semaphore_tracker:UserWarning')

        self.run_process_event = process_event
        self.run_process_completed_event = completed_event
        run_id = self.request_json.get("end_point_id")

        try:
            MLOpsUtils.set_ntp_offset(self.ntp_offset)

            self.setup_client_mqtt_mgr()

            self.run_impl()
        except RunnerError:
            logging.info("Runner stopped.")
            self.mlops_metrics.report_server_training_status(
                self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_KILLED,
                is_from_model=True, edge_id=self.edge_id)
        except RunnerCompletedError:
            logging.info("Runner completed.")
        except Exception as e:
            logging.error("Runner exits with exceptions.")
            logging.error(traceback.format_exc())
            logging.error(e)
            self.mlops_metrics.report_server_training_status(
                self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED,
                is_from_model=True, edge_id=self.edge_id)
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, self.edge_id)
            if self.mlops_metrics is not None:
                self.mlops_metrics.stop_sys_perf()
            time.sleep(3)
            sys.exit(1)
        finally:
            logging.info("Release resources.")
            MLOpsRuntimeLogDaemon.get_instance(self.args).stop_log_processor(run_id, self.edge_id)
            if self.mlops_metrics is not None:
                self.mlops_metrics.stop_sys_perf()
            time.sleep(3)
            if not self.run_as_cloud_server:
                self.release_client_mqtt_mgr()

    def parse_model_run_params(self, running_json):
        run_id = running_json["end_point_id"]
        end_point_name = running_json["end_point_name"]
        token = running_json["token"]
        user_id = running_json["user_id"]
        user_name = running_json["user_name"]
        device_ids = running_json["device_ids"]
        device_objs = running_json["device_objs"]

        model_config = running_json["model_config"]
        model_name = model_config["model_name"]
        model_id = model_config["model_id"]
        model_storage_url = model_config["model_storage_url"]
        scale_min = model_config.get("instance_scale_min", 0)
        scale_max = model_config.get("instance_scale_max", 0)
        inference_engine = model_config.get("inference_engine", 0)
        model_is_from_open = model_config["is_from_open"]
        inference_end_point_id = run_id
        use_gpu = "gpu"  # TODO: Get GPU from device infos
        memory_size = "256m"  # TODO: Get Memory size for each instance
        model_version = model_config["model_version"]        
        model_config_parameters = running_json.get("parameters", {})

        inference_port = model_config_parameters.get("server_internal_port",    # Internal port is for the gateway
                                                     ServerConstants.MODEL_INFERENCE_DEFAULT_PORT)
        inference_port_external = model_config_parameters.get("server_external_port", inference_port)

        return run_id, end_point_name, token, user_id, user_name, device_ids, device_objs, model_config, model_name, \
            model_id, model_storage_url, scale_min, scale_max, inference_engine, model_is_from_open, \
            inference_end_point_id, use_gpu, memory_size, model_version, inference_port

    def inference_run(self):
        # run_id, end_point_name, token, user_id, user_name, device_ids, device_objs, model_config, model_name, \
        #     model_id, model_storage_url, scale_min, scale_max, inference_engine, model_is_from_open, \
        #     inference_end_point_id, use_gpu, memory_size, model_version, inference_port = self.parse_model_run_params(self.request_json)
        #
        # inference_server = FedMLModelServingServer(self.args,
        #                                            end_point_name,
        #                                            model_name,
        #                                            model_version,
        #                                            inference_request=self.request_json)
        # inference_server.run()
        pass

    def run_impl(self):
        run_id, end_point_name, token, user_id, user_name, device_ids, device_objs, model_config, model_name, \
            model_id, model_storage_url, scale_min, scale_max, inference_engine, model_is_from_open, \
            inference_end_point_id, use_gpu, memory_size, model_version, inference_port = self.parse_model_run_params(
            self.request_json)

        logging.info("model deployment request: {}".format(self.request_json))

        # Initiate an FedMLInferenceServer object which the request will be forwarded to
        # server_runner = FedMLServerRunner(
        #     self.args, run_id=self.run_id, request_json=self.request_json, agent_config=self.agent_config
        # )
        # inference_process = Process(target=server_runner.inference_run)
        # inference_process.start()

        logging.info("send deployment stages...")

        self.mlops_metrics.report_sys_perf(self.args, self.agent_config["mqtt_config"], run_id=run_id)

        self.check_runner_stop_event()

        # Send stage: MODEL_DEPLOYMENT_STAGE4 = "ForwardRequest2Slave"
        self.send_deployment_stages(self.run_id, model_name, model_id,
                                    "",
                                    ServerConstants.MODEL_DEPLOYMENT_STAGE4["index"],
                                    ServerConstants.MODEL_DEPLOYMENT_STAGE4["text"],
                                    ServerConstants.MODEL_DEPLOYMENT_STAGE4["text"])

        self.args.run_id = self.run_id
        MLOpsRuntimeLog.get_instance(self.args).init_logs(log_level=logging.INFO)

        # report server running status
        logging.info("report deployment status...")
        self.check_runner_stop_event()
        self.mlops_metrics.report_server_training_status(
            run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_STARTING,
            is_from_model=True, running_json=json.dumps(self.request_json), edge_id=self.edge_id)
        self.send_deployment_status(self.run_id, end_point_name,
                                    model_name, "",
                                    ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYING)

        # start unified inference server
        self.start_device_inference_gateway(
            run_id, end_point_name, model_id, model_name, model_version, inference_port=inference_port)

        # start inference monitor server
        self.stop_device_inference_monitor(run_id, end_point_name, model_id, model_name, model_version)
        self.start_device_inference_monitor(run_id, end_point_name, model_id, model_name, model_version)

        # Changed the status to "IDLE"
        self.mlops_metrics.broadcast_server_training_status(
            run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED,
            is_from_model=True, edge_id=self.edge_id)

        # forward deployment request to slave devices
        logging.info("send the model inference request to slave devices...")
        self.check_runner_stop_event()

        # handle "op:add" && "op:remove"
        devices_sent_add_or_remove_msg = self.send_deployment_start_request_to_edges()

        # handle "op:update"
        devices_sent_update_remove_msg = self.send_first_scroll_update_msg()

        if len(devices_sent_add_or_remove_msg) == 0 and len(devices_sent_update_remove_msg) == 0:
            # No device is added or removed, and no device is updated or removed
            ip = self.get_ip_address(self.request_json)
            master_port = os.getenv("FEDML_MASTER_PORT", None)
            if master_port is not None:
                inference_port = int(master_port)
            model_inference_port = inference_port
            if ip.startswith("http://") or ip.startswith("https://"):
                model_inference_url = "{}/api/v1/predict".format(ip)
            else:
                model_inference_url = "http://{}:{}/api/v1/predict".format(ip, model_inference_port)

            self.set_runner_completed_event(run_id)

            self.send_deployment_status(run_id, end_point_name,
                                        model_name,
                                        model_inference_url,
                                        ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED)
            return

        while True:
            self.check_runner_stop_event()
            time.sleep(3)

    def check_runner_stop_event(self):
        if self.run_process_event is not None and self.run_process_event.is_set():
            logging.info("Received stopping event.")
            raise RunnerError("Runner stopped")

        if self.run_process_completed_event is not None and self.run_process_completed_event.is_set():
            logging.info("Received completed event.")
            raise RunnerCompletedError("Runner completed")

    def start_device_inference_gateway(
            self, run_id, end_point_name, model_id,
            model_name, model_version, inference_port=ServerConstants.MODEL_INFERENCE_DEFAULT_PORT):
        # start unified inference server
        running_model_name = ServerConstants.get_running_model_name(end_point_name,
                                                                    model_name, model_version, run_id, model_id)
        python_program = get_python_program()
        master_port = os.getenv("FEDML_MASTER_PORT", None)
        if master_port is not None:
            inference_port = int(master_port)
        if not ServerConstants.is_running_on_k8s():
            logging.info(f"start the model inference gateway, end point {run_id}, "
                         f"model name {model_name} at port {inference_port}...")
            self.check_runner_stop_event()

            use_mqtt_inference = os.getenv("FEDML_USE_MQTT_INFERENCE", "False")
            use_mqtt_inference = True if use_mqtt_inference.lower() == 'true' else False
            use_worker_gateway = os.getenv("FEDML_USE_WORKER_GATEWAY", "False")
            use_worker_gateway = True if use_worker_gateway.lower() == 'true' else False
            inference_gw_cmd = "fedml.computing.scheduler.model_scheduler.device_model_inference:api"
            inference_gateway_pids = RunProcessUtils.get_pid_from_cmd_line(inference_gw_cmd)
            if inference_gateway_pids is None or len(inference_gateway_pids) <= 0:
                cur_dir = os.path.dirname(__file__)
                fedml_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(cur_dir)))
                connect_str = "@FEDML@"
                ext_info = sys_utils.random1(
                    self.agent_config["mqtt_config"]["BROKER_HOST"] + connect_str +
                    str(self.agent_config["mqtt_config"]["BROKER_PORT"]) + connect_str +
                    self.agent_config["mqtt_config"]["MQTT_USER"] + connect_str +
                    self.agent_config["mqtt_config"]["MQTT_PWD"] + connect_str +
                    str(self.agent_config["mqtt_config"]["MQTT_KEEPALIVE"]), "FEDML@9999GREAT")
                self.inference_gateway_process = ServerConstants.exec_console_with_script(
                    "REDIS_ADDR=\"{}\" REDIS_PORT=\"{}\" REDIS_PASSWORD=\"{}\" "
                    "END_POINT_NAME=\"{}\" "
                    "MODEL_NAME=\"{}\" MODEL_VERSION=\"{}\" MODEL_INFER_URL=\"{}\" VERSION=\"{}\" "
                    "USE_MQTT_INFERENCE={} USE_WORKER_GATEWAY={} EXT_INFO={} "
                    "{} -m uvicorn {} --host 0.0.0.0 --port {} --reload --reload-delay 3 --reload-dir {} "
                    "--log-level critical".format(
                        self.redis_addr, self.redis_port, self.redis_password,
                        end_point_name,
                        model_name, model_version, "", self.args.version,
                        use_mqtt_inference, use_worker_gateway, ext_info,
                        python_program, inference_gw_cmd, str(inference_port), fedml_base_dir
                    ),
                    should_capture_stdout=False,
                    should_capture_stderr=False
                )

    def start_device_inference_monitor(self, run_id, end_point_name,
                                       model_id, model_name, model_version, check_stopped_event=True):
        # start inference monitor server
        logging.info(f"start the model inference monitor, end point {run_id}, model name {model_name}...")
        if check_stopped_event:
            self.check_runner_stop_event()
        run_id_str = str(run_id)
        pip_source_dir = os.path.dirname(__file__)
        monitor_file = os.path.join(pip_source_dir, "device_model_monitor.py")
        python_program = get_python_program()
        running_model_name = ServerConstants.get_running_model_name(end_point_name,
                                                                    model_name, model_version, run_id, model_id)
        self.monitor_process = ServerConstants.exec_console_with_shell_script_list(
            [
                python_program,
                monitor_file,
                "-v",
                self.args.version,
                "-ep",
                run_id_str,
                "-epn",
                str(end_point_name),
                "-mi",
                str(model_id),
                "-mn",
                model_name,
                "-mv",
                model_version,
                "-iu",
                "infer_url",
                "-ra",
                self.redis_addr,
                "-rp",
                self.redis_port,
                "-rpw",
                self.redis_password
            ],
            should_capture_stdout=False,
            should_capture_stderr=False
        )

    def stop_device_inference_monitor(self, run_id, end_point_name, model_id, model_name, model_version):
        # stop inference monitor server
        logging.info(f"stop the model inference monitor, end point {run_id}, model name {model_name}...")
        sys_utils.cleanup_model_monitor_processes(run_id, end_point_name,
                                                  model_id, model_name, model_version)

    def cleanup_run_when_finished(self):
        logging.info("Cleanup run successfully when finished.")

        self.mlops_metrics.broadcast_server_training_status(
            self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED,
            is_from_model=True, edge_id=self.edge_id
        )

        try:
            self.mlops_metrics.stop_sys_perf()
        except Exception as ex:
            pass

        time.sleep(1)

        try:
            local_package_path = ServerConstants.get_package_download_dir()
            for package_file in listdir(local_package_path):
                if os.path.basename(package_file).startswith("run_" + str(self.run_id)):
                    shutil.rmtree(os.path.join(local_package_path, package_file), ignore_errors=True)
        except Exception as e:
            pass

    def cleanup_run_when_starting_failed(self):
        logging.info("Cleanup run successfully when starting failed.")

        self.mlops_metrics.broadcast_server_training_status(
            self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED,
            is_from_model=True, edge_id=self.edge_id)

        try:
            self.mlops_metrics.stop_sys_perf()
        except Exception as ex:
            pass

        time.sleep(1)

        try:
            local_package_path = ServerConstants.get_package_download_dir()
            for package_file in listdir(local_package_path):
                if os.path.basename(package_file).startswith("run_" + str(self.run_id)):
                    shutil.rmtree(os.path.join(local_package_path, package_file), ignore_errors=True)
        except Exception as e:
            pass
    
    def cleanup_run_when_deploy_failed(self):
        topic = f"model_ops/model_device/delete_deployment/{self.edge_id}"
        self.callback_delete_deployment(topic, payload=json.dumps(self.request_json))

    def callback_deployment_result_message(self, topic=None, payload=None):
        # Save deployment result to local cache
        topic_splits = str(topic).split('/')
        device_id = topic_splits[-1]
        payload_json = json.loads(payload)
        end_point_id = payload_json["end_point_id"]
        end_point_name = payload_json["end_point_name"]
        model_id = payload_json["model_id"]
        model_name = payload_json["model_name"]
        model_version = payload_json["model_version"]
        model_status = payload_json["model_status"]
        replica_no = payload_json.get("replica_no", None)  # Idx start from 1
        run_id_str = str(end_point_id)

        # Set redis + sqlite deployment result
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)

        if model_status == ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DELETED:
            FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                delete_deployment_result_with_device_id_and_replica_no(
                    end_point_id, end_point_name, model_name, device_id, replica_no)
        elif model_status == ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED:
            # add or update
            FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                set_deployment_result(end_point_id, end_point_name,
                                      model_name, model_version,
                                      device_id, payload, replica_no)

            # Note: To display the result in the UI, we need to save successful deployment result to the database
            self.model_runner_mapping[run_id_str].deployed_replica_payload = copy.deepcopy(payload_json)
        else:
            if model_status != ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED:
                logging.error(f"Unsupported model status {model_status}.")
            self.send_deployment_status(
                end_point_id, end_point_name, payload_json["model_name"], "",
                ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED)

        # Notify the replica number controller
        (self.model_runner_mapping[run_id_str].
         replica_controller.callback_update_curr_replica_num_state(device_id, replica_no, model_status))

        # Notify the replica version controller, which might trigger the next rolling update
        self.send_next_scroll_update_msg(device_id, replica_no)

        # Update the global deployment result mapping
        if self.slave_deployment_results_mapping.get(run_id_str, None) is None:
            self.slave_deployment_results_mapping[run_id_str] = dict()
        self.slave_deployment_results_mapping[run_id_str][str(device_id)] = model_status

        logging.info("callback_deployment_result_message: topic {}, payload {}, mapping {}.".format(
            topic, payload, self.slave_deployment_results_mapping[run_id_str]))

        request_json = self.running_request_json.get(run_id_str, None)
        if request_json is None:
            logging.error(f"The endpoint {end_point_id} is not running.")
            self.send_deployment_status(
                end_point_id, end_point_name, payload_json["model_name"], "",
                ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED)
            return

        # Wait for all replica's result, not device-level
        if (self.model_runner_mapping[run_id_str].replica_controller.is_all_replica_num_reconciled() and
                self.model_runner_mapping[run_id_str].replica_controller.is_all_replica_version_reconciled()):
            '''
            When all the devices have finished the add / delete / update operation
            '''
            # 1. We should generate one unified inference api
            # Note that here we use the gateway port instead of the inference port that is used by the slave device
            model_config_parameters = request_json["parameters"]
            inference_port = model_config_parameters.get("server_internal_port",
                                                         ServerConstants.MODEL_INFERENCE_DEFAULT_PORT)
            inference_port_external = model_config_parameters.get("server_external_port", inference_port)
            ip = self.get_ip_address(request_json)

            if ip.startswith("http://") or ip.startswith("https://"):
                model_inference_url = "{}/inference/{}".format(ip, end_point_id)
            else:
                model_inference_url = "http://{}:{}/inference/{}".format(ip, inference_port_external, end_point_id)

            # Send stage: MODEL_DEPLOYMENT_STAGE5 = "StartInferenceIngress"
            self.send_deployment_stages(end_point_id, model_name, model_id,
                                        model_inference_url,
                                        ServerConstants.MODEL_DEPLOYMENT_STAGE5["index"],
                                        ServerConstants.MODEL_DEPLOYMENT_STAGE5["text"],
                                        "inference url: {}".format(model_inference_url))

            # Prepare the result to MLOps
            if self.model_runner_mapping[run_id_str].deployed_replica_payload is not None:
                payload_json = self.model_runner_mapping[run_id_str].deployed_replica_payload
                model_slave_url = payload_json["model_url"]
                payload_json["model_url"] = model_inference_url
                payload_json["port"] = inference_port_external
                token = FedMLModelCache.get_instance(self.redis_addr, self.redis_port).get_end_point_token(
                    end_point_id, end_point_name, model_name)

                model_metadata = payload_json["model_metadata"]
                model_inputs = model_metadata["inputs"]
                ret_inputs = list()
                if "type" in model_metadata and model_metadata["type"] == "default":
                    payload_json["input_json"] = {"end_point_name": end_point_name,
                                                  "model_name": model_name,
                                                  "token": str(token),
                                                  "inputs": model_inputs,
                                                  "outputs": []}
                    payload_json["output_json"] = model_metadata["outputs"]
                else:
                    raise Exception(f"Unsupported model metadata type {model_metadata['type']}")

                self.send_deployment_results_with_payload(end_point_id, end_point_name, payload_json)

                payload_json_saved = payload_json
                payload_json_saved["model_slave_url"] = model_slave_url
                FedMLServerDataInterface.get_instance().save_job_result(end_point_id, self.edge_id,
                                                                        json.dumps(payload_json_saved))
            else:
                # Arrive here because only contains remove ops, so we do not need to update the model metadata
                pass

            FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                set_end_point_activation(end_point_id, end_point_name, True)

            self.send_deployment_status(end_point_id, end_point_name,
                                        payload_json["model_name"],
                                        model_inference_url,
                                        ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED)

            self.slave_deployment_results_mapping[run_id_str] = dict()

            time.sleep(3)
            self.set_runner_completed_event(end_point_id)

    def callback_deployment_status_message(self, topic=None, payload=None):
        # [Deprecated] Merge the logic into callback_deployment_result_message
        logging.info("[Deprecated] callback_deployment_status_message: topic {}, payload {}.".format(
            topic, payload))
        pass

    def send_deployment_start_request_to_edges(self):
        # Iterate through replica_num_diff, both add and replace should be sent to the edge devices
        if "replica_num_diff" not in self.request_json or self.request_json["replica_num_diff"] is None:
            return []

        edge_id_list = []
        for device_id in self.request_json["replica_num_diff"].keys():
            edge_id_list.append(device_id)

        self.request_json["master_node_ip"] = self.get_ip_address(self.request_json)
        should_added_devices = []
        for edge_id in edge_id_list:
            if edge_id == self.edge_id:
                continue
            should_added_devices.append(edge_id)
            # send start deployment request to each device
            self.send_deployment_start_request_to_edge(edge_id)
        return should_added_devices

    def send_deployment_start_request_to_edge(self, edge_id):
        topic_start_deployment = "model_ops/model_device/start_deployment/{}".format(str(edge_id))
        logging.info("start_deployment: send topic " + topic_start_deployment + " to client...")
        self.client_mqtt_mgr.send_message_json(topic_start_deployment, json.dumps(self.request_json))

    def get_ip_address(self, request_json):
        # OPTION 1: Use local ip
        ip = ServerConstants.get_local_ip()

        # OPTION 2: Auto detect public ip
        if "parameters" in request_json and \
                ServerConstants.AUTO_DETECT_PUBLIC_IP in request_json["parameters"] and \
                request_json["parameters"][ServerConstants.AUTO_DETECT_PUBLIC_IP]:
            ip = ServerConstants.get_public_ip()
            logging.info("Auto detect public ip for master: " + ip)

        # OPTION 3: Use user indicated ip
        if self.infer_host is not None and self.infer_host != "127.0.0.1" and self.infer_host != "localhost":
            ip = self.infer_host

        return ip

    def send_deployment_delete_request_to_edges(self, payload, model_msg_object):
        edge_id_list_to_delete = model_msg_object.device_ids

        # Remove the model master node id from the list using index 0
        edge_id_list_to_delete = edge_id_list_to_delete[1:]

        logging.info("Device ids to be deleted: " + str(edge_id_list_to_delete))
        
        for edge_id in edge_id_list_to_delete:
            if edge_id == self.edge_id:
                continue
            # send delete deployment request to each model device
            topic_delete_deployment = "model_ops/model_device/delete_deployment/{}".format(str(edge_id))
            logging.info("delete_deployment: send topic " + topic_delete_deployment + " to client...")
            self.client_mqtt_mgr.send_message_json(topic_delete_deployment, payload)

    def ota_upgrade(self, payload, request_json):
        run_id = request_json["end_point_id"]
        force_ota = False
        ota_version = None

        try:
            parameters = request_json.get("parameters", None)
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

            logging.info(f"Upgrade to version {upgrade_version} ...")

            sys_utils.do_upgrade(self.version, upgrade_version)

            raise Exception("Restarting after upgraded...")

    def callback_start_deployment(self, topic, payload):
        """
        topic: model_ops/model_device/start_deployment/model-agent-device-id
        payload:
        {
          "timestamp": 1671440005119,
          "end_point_id": 4325,
          "token": "FCpWU",
          "state": "STARTING",
          "user_id": "105",
          "user_name": "alex.liang2",
          "device_ids": [
            693
          ],
          "device_objs": [
            {
              "device_id": "0xT3630FW2YM@MacOS.Edge.Device",
              "os_type": "MacOS",
              "id": 693,
              "ip": "1.1.1.1",
              "memory": 1024,
              "cpu": "1.7",
              "gpu": "Nvidia",
              "extra_infos": {}
            }
          ],
          "model_config": {
            "model_name": "image-model",
            "model_id": 111,
            "model_version": "v1",
            "is_from_open": 0,
            "model_storage_url": "https://fedml.s3.us-west-1.amazonaws.com/1666239314792client-package.zip",
            "instance_scale_min": 1,
            "instance_scale_max": 3,
            "inference_engine": "onnx"
          },
          "parameters": {
            "hidden_size": 128,
            "hidden_act": "gelu",
            "initializer_range": 0.02,
            "vocab_size": 30522,
            "hidden_dropout_prob": 0.1,
            "num_attention_heads": 2,
            "type_vocab_size": 2,
            "max_position_embeddings": 512,
            "num_hidden_layers": 2,
            "intermediate_size": 512,
            "attention_probs_dropout_prob": 0.1
          }
        }
        """
        try:
            MLOpsConfigs.fetch_all_configs()
        except Exception as e:
            pass

        # get deployment params
        request_json = json.loads(payload)
        run_id = request_json["end_point_id"]
        end_point_name = request_json["end_point_name"]
        token = request_json["token"]
        user_id = request_json["user_id"]
        user_name = request_json["user_name"]
        device_ids = request_json["device_ids"]
        device_objs = request_json["device_objs"]

        model_config = request_json["model_config"]
        model_name = model_config["model_name"]
        model_id = model_config["model_id"]
        model_storage_url = model_config["model_storage_url"]
        scale_min = model_config.get("instance_scale_min", 0)
        scale_max = model_config.get("instance_scale_max", 0)
        inference_engine = model_config.get("inference_engine", 0)
        inference_end_point_id = run_id

        # Start log processor for current run
        self.args.run_id = run_id
        self.args.edge_id = self.edge_id
        MLOpsRuntimeLog.get_instance(self.args).init_logs()
        MLOpsRuntimeLogDaemon.get_instance(self.args).set_log_source(
            ServerConstants.FEDML_LOG_SOURCE_TYPE_MODEL_END_POINT)
        MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(run_id, self.edge_id)

        logging.info("callback_start_deployment {}".format(payload))

        # self.ota_upgrade(payload, request_json)

        run_id = inference_end_point_id
        self.args.run_id = run_id
        self.run_id = run_id
        request_json["run_id"] = run_id
        self.request_json = request_json
        run_id_str = str(run_id)
        self.running_request_json[run_id_str] = request_json

        self.request_json["master_node_ip"] = self.get_ip_address(self.request_json)

        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)

        # Target status of the devices
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            set_end_point_device_info(request_json["end_point_id"], end_point_name, json.dumps(device_objs))

        usr_indicated_token = self.get_usr_indicated_token(request_json)
        if usr_indicated_token != "":
            logging.info(f"Change Token from{token} to {usr_indicated_token}")
            token = usr_indicated_token
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            set_end_point_token(run_id, end_point_name, model_name, token)

        ServerConstants.save_runner_infos(self.args.device_id + "." + self.args.os_name, self.edge_id, run_id=run_id)

        self.subscribe_slave_devices_message(request_json)

        # Send stage: MODEL_DEPLOYMENT_STAGE1 = "Received"
        self.send_deployment_stages(self.run_id, model_name, model_id,
                                    "",
                                    ServerConstants.MODEL_DEPLOYMENT_STAGE1["index"],
                                    ServerConstants.MODEL_DEPLOYMENT_STAGE1["text"],
                                    "Received request for end point {}".format(run_id))

        # Send stage: MODEL_DEPLOYMENT_STAGE2 = "Initializing"
        self.send_deployment_stages(self.run_id, model_name, model_id,
                                    "",
                                    ServerConstants.MODEL_DEPLOYMENT_STAGE2["index"],
                                    ServerConstants.MODEL_DEPLOYMENT_STAGE2["text"],
                                    ServerConstants.MODEL_DEPLOYMENT_STAGE2["text"])

        ServerConstants.save_runner_infos(self.args.device_id + "." + self.args.os_name, self.edge_id, run_id=run_id)

        if self.run_as_edge_server_and_agent:
            # Replica Controller is per deployment
            replica_controller = FedMLDeviceReplicaController(self.edge_id, self.request_json)
            logging.info(f"Start Diff Replica controller for run {run_id} on edge {self.edge_id}")

            # Prepare num diff
            new_request_with_num_diff = replica_controller.generate_diff_to_request_json()
            self.running_request_json[run_id_str] = new_request_with_num_diff
            request_json = new_request_with_num_diff

            # Prepare version diff
            new_request_with_version_diff = replica_controller.init_first_update_device_replica_mapping()
            self.running_request_json[run_id_str] = new_request_with_version_diff
            request_json = new_request_with_version_diff

            server_runner = FedMLServerRunner(
                self.args, run_id=run_id, request_json=request_json, agent_config=self.agent_config
            )
            server_runner.run_as_edge_server_and_agent = self.run_as_edge_server_and_agent
            server_runner.edge_id = self.edge_id
            server_runner.infer_host = self.infer_host
            server_runner.redis_addr = self.redis_addr
            server_runner.redis_port = self.redis_port
            server_runner.redis_password = self.redis_password
            server_runner.replica_controller = replica_controller

            self.run_process_event_map[run_id_str] = multiprocessing.Event()
            self.run_process_event_map[run_id_str].clear()
            server_runner.run_process_event = self.run_process_event_map[run_id_str]
            self.run_process_completed_event_map[run_id_str] = multiprocessing.Event()
            self.run_process_completed_event_map[run_id_str].clear()
            server_runner.run_process_completed_event = self.run_process_completed_event_map[run_id_str]
            self.model_runner_mapping[run_id_str] = server_runner

            # This subprocess will copy the server_runner and run it, but they are not the same object
            server_process = Process(target=server_runner.run, args=(
                self.run_process_event_map[run_id_str], self.run_process_completed_event_map[run_id_str]
            ))
            server_process.start()
            ServerConstants.save_run_process(run_id, server_process.pid)

            # Send stage: MODEL_DEPLOYMENT_STAGE3 = "StartRunner"
            self.send_deployment_stages(self.run_id, model_name, model_id,
                                        "",
                                        ServerConstants.MODEL_DEPLOYMENT_STAGE3["index"],
                                        ServerConstants.MODEL_DEPLOYMENT_STAGE3["text"],
                                        ServerConstants.MODEL_DEPLOYMENT_STAGE3["text"])

    def send_first_scroll_update_msg(self):
        """
        Replica-level rolling update.
        Delete the record of the replaced device and send the deployment msg to the devices
        """
        if "replica_version_diff" not in self.request_json or self.request_json["replica_version_diff"] is None:
            return []

        first_chunk_dict = self.request_json["replica_version_diff"]

        # Delete the record of the replaced device
        self.delete_device_replica_info_on_master(first_chunk_dict)

        # Send the deployment msg to the devices, (we reuse the start_deployment msg)
        for edge_id in first_chunk_dict.keys():
            if edge_id == self.edge_id:
                continue
            # send start deployment request to each device
            self.send_deployment_start_request_to_edge(edge_id)
        return list(first_chunk_dict.keys())
    
    def delete_device_replica_info_on_master(self, edge_id_replica_no_dict):
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        # Remove the record of the replaced device
        # [Deprecated] deployment status & device info
        # Delete the result in deployment result list in Redis / SQLite
        device_result_list = FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            get_deployment_result_list(self.request_json["end_point_id"], self.request_json["end_point_name"],
                                       self.request_json["model_config"]["model_name"])
        delete_device_result_list = []
        for device_result in device_result_list:
            device_result_dict = json.loads(device_result)
            if (str(device_result_dict["cache_device_id"]) in edge_id_replica_no_dict.keys() and
                    str(device_result_dict["cache_replica_no"]) in
                    edge_id_replica_no_dict[str(device_result_dict["cache_device_id"])]):
                delete_device_result_list.append(device_result)
        
        for delete_item in delete_device_result_list:
            FedMLModelCache.get_instance(self.redis_addr, self.redis_port).delete_deployment_result(
                delete_item, self.request_json["end_point_id"],
                self.request_json["end_point_name"],
                self.request_json["model_config"]["model_name"]
            )
        
        logging.info(f"Deleted the record of the replaced device {delete_device_result_list}")

    def send_next_scroll_update_msg(self, device_id, replica_no):
        if replica_no is None:
            return

        replica_controller = self.model_runner_mapping[str(self.run_id)].replica_controller

        if replica_controller.total_replica_version_diff_num == 0:
            return

        replica_controller.callback_update_updating_window(device_id, replica_no)

        # Decide whether to send the next scroll update
        next_chunk_dict = replica_controller.get_next_chunk_devices_replica()

        replica_controller.curr_replica_updating_window = copy.deepcopy(next_chunk_dict)

        if next_chunk_dict:
            self.request_json["replica_version_diff"] = next_chunk_dict
            self.delete_device_replica_info_on_master(next_chunk_dict)

            # Send the deployment msg to the devices, (we reuse the start_deployment msg)
            for edge_id in next_chunk_dict.keys():
                if edge_id == self.edge_id:
                    continue
                # send start deployment request to each device
                self.send_deployment_start_request_to_edge(edge_id)
        return

    def callback_activate_deployment(self, topic, payload):
        logging.info("callback_activate_deployment: topic = %s, payload = %s" % (topic, payload))

        # Parse payload as the model message object.
        model_msg_object = FedMLModelMsgObject(topic, payload)

        # Get the previous deployment status.
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        endpoint_status = FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            get_end_point_status(model_msg_object.inference_end_point_id)
        if endpoint_status != ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED:
            return

        # Set end point as activated status
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port).set_end_point_activation(
            model_msg_object.inference_end_point_id, model_msg_object.end_point_name, True)

    def callback_deactivate_deployment(self, topic, payload):
        logging.info("callback_deactivate_deployment: topic = %s, payload = %s" % (topic, payload))

        # Parse payload as the model message object.
        model_msg_object = FedMLModelMsgObject(topic, payload)

        # Get the endpoint status
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        endpoint_status = FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            get_end_point_status(model_msg_object.inference_end_point_id)
        if endpoint_status != ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED:
            return

        # Set end point as deactivated status
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port).set_end_point_activation(
            model_msg_object.inference_end_point_id, model_msg_object.model_name, False)

    def set_runner_stopped_event(self, run_id):
        run_id_str = str(run_id)
        server_runner = self.model_runner_mapping.get(run_id_str, None)
        if server_runner is not None:
            if server_runner.run_process_event is not None:
                server_runner.run_process_event.set()
            self.model_runner_mapping.pop(run_id_str)

    def set_runner_completed_event(self, run_id):
        run_id_str = str(run_id)
        server_runner = self.model_runner_mapping.get(run_id_str, None)
        if server_runner is not None:
            if server_runner.run_process_completed_event is not None:
                server_runner.run_process_completed_event.set()
            self.model_runner_mapping.pop(run_id_str)

    def callback_delete_deployment(self, topic, payload):
        logging.info("[Master] callback_delete_deployment")
        # Parse payload as the model message object.
        model_msg_object = FedMLModelMsgObject(topic, payload)

        # Set end point as deactivated status
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            set_end_point_activation(model_msg_object.inference_end_point_id,
                                     model_msg_object.end_point_name, False)
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            delete_end_point(model_msg_object.inference_end_point_id, model_msg_object.end_point_name,
                             model_msg_object.model_name, model_msg_object.model_version)

        self.send_deployment_delete_request_to_edges(payload, model_msg_object)

        self.set_runner_stopped_event(model_msg_object.run_id)

        self.stop_device_inference_monitor(model_msg_object.run_id, model_msg_object.end_point_name,
                                           model_msg_object.model_id, model_msg_object.model_name,
                                           model_msg_object.model_version)

        FedMLServerDataInterface.get_instance().delete_job_from_db(model_msg_object.run_id)
        FedMLModelDatabase.get_instance().delete_deployment_result(
            model_msg_object.run_id, model_msg_object.end_point_name, model_msg_object.model_name,
            model_version=model_msg_object.model_version)
        FedMLModelDatabase.get_instance().delete_deployment_run_info(
            end_point_id=model_msg_object.inference_end_point_id)

    def send_deployment_results_with_payload(self, end_point_id, end_point_name, payload):
        self.send_deployment_results(end_point_id, end_point_name,
                                     payload["model_name"], payload["model_url"],
                                     payload["model_version"], payload["port"],
                                     payload["inference_engine"],
                                     payload["model_metadata"],
                                     payload["model_config"],
                                     payload["input_json"],
                                     payload["output_json"])

    def send_deployment_results(self, end_point_id, end_point_name,
                                model_name, model_inference_url,
                                model_version, inference_port, inference_engine,
                                model_metadata, model_config, input_json, output_json):
        deployment_results_topic_prefix = "model_ops/model_device/return_deployment_result"
        deployment_results_topic = "{}/{}".format(deployment_results_topic_prefix, end_point_id)
        deployment_results_payload = {"end_point_id": end_point_id, "end_point_name": end_point_name,
                                      "model_name": model_name, "model_url": model_inference_url,
                                      "version": model_version, "port": inference_port,
                                      "inference_engine": inference_engine,
                                      "model_metadata": model_metadata,
                                      "model_config": model_config,
                                      "input_json": input_json,
                                      "output_json": output_json,
                                      "timestamp": int(format(time.time_ns() / 1000.0, '.0f'))}
        logging.info(f"[Master] deployment_results_payload is sent to mlops: {deployment_results_payload}")

        self.client_mqtt_mgr.send_message_json(deployment_results_topic, json.dumps(deployment_results_payload))
        self.client_mqtt_mgr.send_message_json(deployment_results_topic_prefix, json.dumps(deployment_results_payload))

    def send_deployment_status(self, end_point_id, end_point_name, model_name, model_inference_url, model_status):
        deployment_status_topic_prefix = "model_ops/model_device/return_deployment_status"
        deployment_status_topic = "{}/{}".format(deployment_status_topic_prefix, end_point_id)
        deployment_status_payload = {"end_point_id": end_point_id, "end_point_name": end_point_name,
                                     "model_name": model_name,
                                     "model_url": model_inference_url,
                                     "model_status": model_status,
                                     "timestamp": int(format(time.time_ns() / 1000.0, '.0f'))}
        logging.info(f"[Master] deployment_status_payload is sent to mlops: {deployment_status_payload}")

        self.client_mqtt_mgr.send_message_json(deployment_status_topic, json.dumps(deployment_status_payload))
        self.client_mqtt_mgr.send_message_json(deployment_status_topic_prefix, json.dumps(deployment_status_payload))

    def send_deployment_stages(self, end_point_id, model_name, model_id, model_inference_url,
                               model_stages_index, model_stages_title, model_stage_detail):
        deployment_stages_topic_prefix = "model_ops/model_device/return_deployment_stages"
        deployment_stages_topic = "{}/{}".format(deployment_stages_topic_prefix, end_point_id)
        deployment_stages_payload = {"model_name": model_name,
                                     "model_id": model_id,
                                     "model_url": model_inference_url,
                                     "end_point_id": end_point_id,
                                     "model_stage_index": model_stages_index,
                                     "model_stage_title": model_stages_title,
                                     "model_stage_detail": model_stage_detail,
                                     "timestamp": int(format(time.time_ns() / 1000.0, '.0f'))}

        self.client_mqtt_mgr.send_message_json(deployment_stages_topic, json.dumps(deployment_stages_payload))
        self.client_mqtt_mgr.send_message_json(deployment_stages_topic_prefix, json.dumps(deployment_stages_payload))

        logging.info(f"-------- Stages has been sent to mlops with stage {model_stages_index} and "
                     f"payload {deployment_stages_payload}")
        time.sleep(2)

    def on_client_mqtt_disconnected(self, mqtt_client_object):
        if self.client_mqtt_lock is None:
            self.client_mqtt_lock = threading.Lock()

        self.client_mqtt_lock.acquire()
        self.client_mqtt_is_connected = False
        self.client_mqtt_lock.release()

        logging.info("on_client_mqtt_disconnected: {}.".format(self.client_mqtt_is_connected))

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
            "FedML_ModelServerAgent_Metrics_@{}@_{}_{}_{}".format(self.user_name, self.args.current_device_id,
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

    def send_deployment_stop_request_to_edges(self, edge_id_list, payload):
        for edge_id in edge_id_list:
            topic_stop_deployment = "model_ops/model_device/stop_deployment/{}".format(str(self.edge_id))
            logging.info("stop_deployment: send topic " + topic_stop_deployment)
            self.client_mqtt_mgr.send_message_json(topic_stop_deployment, payload)

    def send_exit_train_with_exception_request_to_edges(self, edge_id_list, payload):
        for edge_id in edge_id_list:
            topic_exit_train = "flserver_agent/" + str(edge_id) + "/exit_train_with_exception"
            logging.info("exit_train_with_exception: send topic " + topic_exit_train)
            self.client_mqtt_mgr.send_message_json(topic_exit_train, payload)

    def exit_run_with_exception_entry(self):
        try:
            self.setup_client_mqtt_mgr()
            self.exit_run_with_exception()
        except Exception as e:
            self.release_client_mqtt_mgr()
            sys_utils.cleanup_all_fedml_server_login_processes(
                ServerConstants.SERVER_LOGIN_PROGRAM, clean_process_group=False)
            sys.exit(1)
        finally:
            self.release_client_mqtt_mgr()

    def exit_run_with_exception(self):
        logging.info("Exit run successfully.")

        ServerConstants.cleanup_learning_process(self.run_id)
        ServerConstants.cleanup_run_process(self.run_id)

        self.mlops_metrics.report_server_id_status(
            self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED, edge_id=self.edge_id)

        time.sleep(1)

    def callback_exit_train_with_exception(self, topic, payload):
        # logging.info("callback_exit_train_with_exception: topic = %s, payload = %s" % (topic, payload))

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
        if run_id is None:
            logging.info("callback_client_exit_train_with_exception run id is none")
            return

        job = FedMLServerDataInterface.get_instance().get_job_by_id(run_id)
        if job is not None and job.running_json is not None and job.running_json != "":
            job_json_obj = json.loads(job.running_json)
            edge_ids = job_json_obj.get("edgeids", None)

            self.mlops_metrics.broadcast_server_training_status(
                run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED,
                is_from_model=True, edge_id=edge_id)

            self.send_exit_train_with_exception_request_to_edges(edge_ids, job.running_json)

            self.exit_run_with_exception()

    def callback_runner_id_status(self, topic, payload):
        logging.info("callback_runner_id_status: topic = %s, payload = %s" % (topic, payload))

        request_json = json.loads(payload)
        is_retain = request_json.get("is_retain", False)
        if is_retain:
            return
        run_id = request_json["run_id"]
        status = request_json["status"]
        edge_id = request_json["edge_id"]
        run_id_str = str(run_id)

        if (
                status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED
                or status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED
        ):
            # Stop server with multiprocessing mode
            stop_request_json = self.running_request_json.get(run_id_str, None)
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

    def cleanup_client_with_status(self):
        if self.run_status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED:
            logging.info("received to finished status.")
            self.cleanup_run_when_finished()
        elif self.run_status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED:
            logging.info("received to failed status.")
            self.cleanup_run_when_starting_failed()

    def callback_report_current_status(self, topic, payload):
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

    def bind_account_and_device_id(self, url, account_id, device_id, os_name):
        role = ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_ON_PREMISE_MASTER_INDEX]
        if self.run_as_edge_server_and_agent:
            role = ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_ON_PREMISE_MASTER_INDEX]
        elif self.run_as_cloud_agent:
            role = ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_FEDML_CLOUD_MASTER_INDEX]
        elif self.run_as_cloud_server:
            role = ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_INFERENCE_INSTANCE_INDEX]

        ip = requests.get('https://checkip.amazonaws.com').text.strip()
        fedml_ver, exec_path, os_ver, cpu_info, python_ver, torch_ver, mpi_installed, \
            cpu_usage, available_mem, total_mem, gpu_info, gpu_available_mem, gpu_total_mem, \
            gpu_count, gpu_vendor, cpu_count, gpu_device_name = get_sys_runner_info()
        host_name = sys_utils.get_host_name()
        json_params = {
            "accountid": account_id,
            "deviceid": device_id,
            "type": os_name,
            "state": ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE,
            "status": ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE,
            "processor": cpu_info,
            "core_type": cpu_info,
            "network": "",
            "role": role,
            "os_ver": os_ver,
            "memory": total_mem,
            "ip": ip,
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
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    url, json=json_params, verify=True,
                    headers={"content-type": "application/json", "Connection": "close"}
                )
        else:
            response = requests.post(url, json=json_params, headers={"Connection": "close"})
        edge_id = -1
        user_name = None
        extra_url = None
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
                if status_code == SchedulerConstants.BINDING_ACCOUNT_NOT_EXIST_ERROR:
                    raise SystemExit(SchedulerConstants.BINDING_ACCOUNT_NOT_EXIST_ERROR)
                print(f"Binding to MLOps with response.status_code = {response.status_code}, "
                      f"response.content: {response.content}")
                return -1, None, None
        return edge_id, user_name, extra_url

    def fetch_configs(self):
        return MLOpsConfigs.fetch_all_configs()

    def send_agent_active_msg(self):
        active_topic = "flserver_agent/active"
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

    def subscribe_slave_devices_message(self, request_json):
        if request_json is None:
            return
        run_id = request_json["run_id"]
        edge_id_list = request_json["device_ids"]
        logging.info("Edge ids: " + str(edge_id_list))
        for edge_id in edge_id_list:
            if str(edge_id) == str(self.edge_id):
                continue
            # subscribe deployment result message for each model device
            deployment_results_topic = "model_device/model_device/return_deployment_result/{}".format(edge_id)
            self.mqtt_mgr.add_message_listener(deployment_results_topic, self.callback_deployment_result_message)
            self.mqtt_mgr.subscribe_msg(deployment_results_topic)

            # subscribe deployment status message for each model device
            deployment_status_topic = "model_device/model_device/return_deployment_status/{}".format(edge_id)
            self.mqtt_mgr.add_message_listener(deployment_status_topic, self.callback_deployment_status_message)
            self.mqtt_mgr.subscribe_msg(deployment_status_topic)

            logging.info("subscribe device messages {}, {}".format(
                deployment_results_topic, deployment_status_topic))

    def on_agent_mqtt_connected(self, mqtt_client_object):
        # The MQTT message topic format is as follows: <sender>/<receiver>/<action>

        # Setup MQTT message listener for starting deployment
        server_agent_id = self.edge_id
        topic_start_deployment = "model_ops/model_device/start_deployment/{}".format(str(self.edge_id))
        self.mqtt_mgr.add_message_listener(topic_start_deployment, self.callback_start_deployment)

        # Setup MQTT message listener for activating deployment
        topic_activate_deployment = "model_ops/model_device/activate_deployment/{}".format(str(self.edge_id))
        self.mqtt_mgr.add_message_listener(topic_activate_deployment, self.callback_activate_deployment)

        # Setup MQTT message listener for deactivating deployment
        topic_deactivate_deployment = "model_ops/model_device/deactivate_deployment/{}".format(str(self.edge_id))
        self.mqtt_mgr.add_message_listener(topic_deactivate_deployment, self.callback_deactivate_deployment)

        # Setup MQTT message listener for delete deployment
        topic_delete_deployment = "model_ops/model_device/delete_deployment/{}".format(str(self.edge_id))
        self.mqtt_mgr.add_message_listener(topic_delete_deployment, self.callback_delete_deployment)

        # Setup MQTT message listener for server status switching
        topic_server_status = "fl_server/flserver_agent_" + str(server_agent_id) + "/status"
        self.mqtt_mgr.add_message_listener(topic_server_status, self.callback_runner_id_status)

        # Setup MQTT message listener to report current device status.
        topic_report_status = "mlops/report_device_status"
        self.mqtt_mgr.add_message_listener(topic_report_status, self.callback_report_current_status)

        # Setup MQTT message listener to OTA messages from the MLOps.
        topic_ota_msg = "mlops/flserver_agent_" + str(server_agent_id) + "/ota"
        self.mqtt_mgr.add_message_listener(topic_ota_msg, self.callback_server_ota_msg)

        # Subscribe topics for starting train, stopping train and fetching client status.
        mqtt_client_object.subscribe(topic_start_deployment, qos=2)
        mqtt_client_object.subscribe(topic_activate_deployment, qos=2)
        mqtt_client_object.subscribe(topic_deactivate_deployment, qos=2)
        mqtt_client_object.subscribe(topic_delete_deployment, qos=2)
        mqtt_client_object.subscribe(topic_server_status, qos=2)
        mqtt_client_object.subscribe(topic_report_status, qos=2)
        mqtt_client_object.subscribe(topic_ota_msg, qos=2)

        self.subscribed_topics.clear()
        self.subscribed_topics.append(topic_start_deployment)
        self.subscribed_topics.append(topic_activate_deployment)
        self.subscribed_topics.append(topic_deactivate_deployment)
        self.subscribed_topics.append(topic_delete_deployment)
        self.subscribed_topics.append(topic_server_status)
        self.subscribed_topics.append(topic_report_status)
        self.subscribed_topics.append(topic_ota_msg)

        self.endpoint_sync_protocol = FedMLEndpointSyncProtocol(agent_config=self.agent_config, mqtt_mgr=self.mqtt_mgr)
        self.endpoint_sync_protocol.setup_listener_for_sync_device_info(self.edge_id)

        # Broadcast the first active message.
        self.send_agent_active_msg()

        # Echo results
        # print("\n\nCongratulations, your device is connected to the FedML MLOps platform successfully!")
        # print(
        #     "Your FedML Edge ID is " + str(self.edge_id) + ", unique device ID is "
        #     + str(self.unique_device_id)
        #     + "\n"
        # )

        MLOpsRuntimeLog.get_instance(self.args).init_logs(log_level=logging.INFO)

    def on_agent_mqtt_disconnected(self, mqtt_client_object):
        MLOpsStatus.get_instance().set_server_agent_status(
            self.edge_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE
        )

    def recover_inference_and_monitor(self):
        try:
            history_jobs = FedMLServerDataInterface.get_instance().get_history_jobs()
            for job in history_jobs.job_list:
                if job.running_json is None:
                    continue

                if job.deployment_result == "":
                    continue

                run_id, end_point_name, token, user_id, user_name, device_ids, device_objs, model_config, model_name, \
                    model_id, model_storage_url, scale_min, scale_max, inference_engine, model_is_from_open, \
                    inference_end_point_id, use_gpu, memory_size, model_version, inference_port = \
                    self.parse_model_run_params(json.loads(job.running_json))

                FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
                is_activated = FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                    get_end_point_activation(run_id)
                if not is_activated:
                    continue

                self.start_device_inference_gateway(run_id, end_point_name, model_id, model_name, model_version,
                                                    inference_port=inference_port)

                self.stop_device_inference_monitor(run_id, end_point_name, model_id, model_name, model_version)
                self.start_device_inference_monitor(run_id, end_point_name, model_id, model_name, model_version)
        except Exception as e:
            logging.info("recover inference and monitor: {}".format(traceback.format_exc()))

    def recover_start_deployment_msg_after_upgrading(self):
        try:
            current_job = FedMLServerDataInterface.get_instance().get_current_job()
            if current_job is not None and \
                    current_job.status == ServerConstants.MSG_MLOPS_SERVER_STATUS_UPGRADING:
                FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
                is_activated = FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                    get_end_point_activation(current_job.job_id)
                if not is_activated:
                    return
                logging.info("start deployment after upgrading.")
                topic_start_deployment = "model_ops/model_device/start_deployment/{}".format(str(self.edge_id))
                self.callback_start_deployment(topic_start_deployment, current_job.running_json)
        except Exception as e:
            logging.info("recover starting deployment message after upgrading: {}".format(traceback.format_exc()))

    def setup_agent_mqtt_connection(self, service_config):
        # Setup MQTT connection
        self.mqtt_mgr = MqttManager(
            service_config["mqtt_config"]["BROKER_HOST"],
            service_config["mqtt_config"]["BROKER_PORT"],
            service_config["mqtt_config"]["MQTT_USER"],
            service_config["mqtt_config"]["MQTT_PWD"],
            service_config["mqtt_config"]["MQTT_KEEPALIVE"],
            "FedML_ModelServerAgent_Daemon_@" + self.user_name + "@_" + self.args.current_device_id + str(uuid.uuid4()),
            "flserver_agent/last_will_msg",
            json.dumps({"ID": self.edge_id, "status": ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE})
        )
        self.agent_config = service_config

        # Init local database
        FedMLServerDataInterface.get_instance().create_job_table()
        try:
            FedMLModelDatabase.get_instance().set_database_base_dir(ServerConstants.get_database_dir())
            FedMLModelDatabase.get_instance().create_table()
        except Exception as e:
            pass

        server_api_cmd = "fedml.computing.scheduler.model_scheduler.device_server_api:api"
        server_api_pids = RunProcessUtils.get_pid_from_cmd_line(server_api_cmd)
        if server_api_pids is None or len(server_api_pids) <= 0:
            # Start local API services
            cur_dir = os.path.dirname(__file__)
            fedml_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(cur_dir)))
            python_program = get_python_program()
            self.local_api_process = ServerConstants.exec_console_with_script(
                "{} -m uvicorn {} --host 0.0.0.0 --port {} --reload --reload-delay 3 --reload-dir {} "
                "--log-level critical".format(
                    python_program, server_api_cmd, ServerConstants.LOCAL_SERVER_API_PORT,
                    fedml_base_dir
                ),
                should_capture_stdout=False,
                should_capture_stderr=False
            )
            # if self.local_api_process is not None and self.local_api_process.pid is not None:
            #     print(f"Model master local API process id {self.local_api_process.pid}")

        self.recover_inference_and_monitor()

        # MLOpsRuntimeLogDaemon.get_instance(self.args).stop_all_log_processor()

        # Setup MQTT connected listener
        self.mqtt_mgr.add_connected_listener(self.on_agent_mqtt_connected)
        self.mqtt_mgr.add_disconnected_listener(self.on_agent_mqtt_disconnected)
        self.mqtt_mgr.connect()

        self.setup_client_mqtt_mgr()
        self.mlops_metrics.report_server_training_status(
            self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE,
            is_from_model=True, edge_id=self.edge_id)
        MLOpsStatus.get_instance().set_server_agent_status(
            self.edge_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE
        )

        self.recover_start_deployment_msg_after_upgrading()

    def stop_agent(self):
        if self.run_process_event is not None:
            self.run_process_event.set()

        if self.mqtt_mgr is not None:
            try:
                for topic in self.subscribed_topics:
                    self.mqtt_mgr.unsubscribe_msg(topic)
            except Exception as e:
                pass

            self.mqtt_mgr.loop_stop()
            self.mqtt_mgr.disconnect()

        self.release_client_mqtt_mgr()

    def start_agent_mqtt_loop(self, should_exit_sys=True):
        # Start MQTT message loop
        try:
            self.mqtt_mgr.loop_forever()
        except Exception as e:
            if str(e) == "Restarting after upgraded...":
                logging.info("Restarting after upgraded...")
            else:
                print("Server tracing: {}".format(traceback.format_exc()))
        finally:
            self.stop_agent()
            if should_exit_sys:
                pass
                """
                    # Deprecated, will kill the process by the parent process.
                    time.sleep(5)
                    sys_utils.cleanup_all_fedml_server_login_processes(
                    ServerConstants.SERVER_LOGIN_PROGRAM, clean_process_group=False)
                    sys.exit(1)
                """

