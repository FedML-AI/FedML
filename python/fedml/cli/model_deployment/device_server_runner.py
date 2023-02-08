import base64
import copy
import json
import logging
import platform
from urllib.parse import urlparse

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
from ...cli.model_deployment.device_client_constants import ClientConstants
from ...cli.model_deployment.device_server_constants import ServerConstants

from ...core.mlops.mlops_metrics import MLOpsMetrics

from ...core.mlops.mlops_configs import MLOpsConfigs
from ...core.mlops.mlops_runtime_log_daemon import MLOpsRuntimeLogDaemon
from ...core.mlops.mlops_status import MLOpsStatus
from ..comm_utils.sys_utils import get_sys_runner_info, get_python_program
from .device_model_cache import FedMLModelCache
from .device_model_msg_object import FedMLModelMsgObject


class FedMLServerRunner:
    FEDML_CLOUD_SERVER_PREFIX = "fedml-server-run-"
    FEDML_BOOTSTRAP_RUN_OK = "[FedML]Bootstrap Finished"

    def __init__(self, args, run_id=0, request_json=None, agent_config=None, edge_id=0):
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

        self.FEDML_DYNAMIC_CONSTRAIN_VARIABLES = {}

        self.mlops_metrics = None
        self.client_agent_active_list = dict()
        self.server_active_list = dict()
        self.run_status = None
        self.infer_host = "127.0.0.1"
        self.redis_addr = "local"
        self.redis_port = "6379"
        self.redis_password = "fedml_default"

        self.slave_deployment_statuses_mapping = {}
        self.slave_deployment_results_mapping = {}

    def build_dynamic_constrain_variables(self, run_id, run_config):
        pass

    def unzip_file(self, zip_file, unzip_file_path):
        result = False
        if zipfile.is_zipfile(zip_file):
            with zipfile.ZipFile(zip_file, "r") as zipf:
                zipf.extractall(unzip_file_path)
                result = True

        return result

    def retrieve_and_unzip_package(self, package_name, package_url):
        local_package_path = ServerConstants.get_model_package_dir()
        if not os.path.exists(local_package_path):
            os.makedirs(local_package_path)
        local_package_file = "{}.zip".format(os.path.join(local_package_path, package_name))
        if not os.path.exists(local_package_file):
            urllib.request.urlretrieve(package_url, local_package_file)
        unzip_package_path = ServerConstants.get_model_dir()
        self.fedml_packages_base_dir = unzip_package_path
        try:
            shutil.rmtree(
                os.path.join(unzip_package_path, package_name), ignore_errors=True
            )
        except Exception as e:
            pass
        self.unzip_file(local_package_file, unzip_package_path)
        unzip_package_path = os.path.join(unzip_package_path, package_name)
        return unzip_package_path

    def update_local_fedml_config(self, run_id, run_config):
        model_config = run_config
        model_name = model_config["model_name"]
        model_storage_url = model_config["model_storage_url"]
        scale_min = model_config["instance_scale_min"]
        scale_max = model_config["instance_scale_max"]
        inference_engine = model_config["inference_engine"]
        inference_end_point_id = run_id

        # Copy config file from the client
        unzip_package_path = self.retrieve_and_unzip_package(
            model_name, model_storage_url
        )
        fedml_local_config_file = os.path.join(unzip_package_path, "model_config.yaml")

        # Load the above config to memory
        package_conf_object = {}
        if os.path.exists(fedml_local_config_file):
            package_conf_object = load_yaml_config(fedml_local_config_file)

        return unzip_package_path, package_conf_object

    def build_dynamic_args(self, run_config, package_conf_object, base_dir):
        pass

    def run(self):
        run_id = self.request_json["end_point_id"]
        token = self.request_json["token"]
        user_id = self.request_json["user_id"]
        user_name = self.request_json["user_name"]
        device_ids = self.request_json["device_ids"]
        device_objs = self.request_json["device_objs"]

        model_config = self.request_json["model_config"]
        model_name = model_config["model_name"]
        model_id = model_config["model_id"]
        model_storage_url = model_config["model_storage_url"]
        scale_min = model_config["instance_scale_min"]
        scale_max = model_config["instance_scale_max"]
        inference_engine = model_config["inference_engine"]
        model_is_from_open = model_config["is_from_open"]
        inference_end_point_id = run_id
        use_gpu = "gpu"  # TODO: Get GPU from device infos
        memory_size = "256m"  # TODO: Get Memory size for each instance
        model_version = "v1"

        logging.info("Model deployment request: {}".format(self.request_json))

        # set mqtt connection for client
        self.setup_client_mqtt_mgr()

        # Send stage: MODEL_DEPLOYMENT_STAGE4 = "ForwardRequest2Slave"
        self.send_deployment_stages(self.run_id, model_name, model_id,
                                    "",
                                    ServerConstants.MODEL_DEPLOYMENT_STAGE4["index"],
                                    ServerConstants.MODEL_DEPLOYMENT_STAGE4["text"],
                                    ServerConstants.MODEL_DEPLOYMENT_STAGE4["text"])

        self.args.run_id = self.run_id
        MLOpsRuntimeLog.get_instance(self.args).init_logs(show_stdout_log=True)

        # report server running status
        self.mlops_metrics.report_server_training_status(run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        self.send_deployment_status(self.run_id, model_name, "",
                                    ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYING)

        # start unified inference server
        running_model_name = ServerConstants.get_running_model_name(run_id, model_id,
                                                                    model_name, model_version)
        if not ServerConstants.is_running_on_k8s():
            process = ServerConstants.exec_console_with_script(
                "REDIS_ADDR=\"{}\" REDIS_PORT=\"{}\" REDIS_PASSWORD=\"{}\" "
                "END_POINT_ID=\"{}\" MODEL_ID=\"{}\" "
                "MODEL_NAME=\"{}\" MODEL_VERSION=\"{}\" MODEL_INFER_URL=\"{}\" VERSION=\"{}\" "
                "uvicorn fedml.cli.model_deployment.device_model_inference:api --host 0.0.0.0 --port {} --reload".format(
                    self.redis_addr, self.redis_port, self.redis_password,
                    str(self.run_id), str(model_id),
                    running_model_name, model_version, "", self.args.version,
                    str(ServerConstants.MODEL_INFERENCE_DEFAULT_PORT)),
                should_capture_stdout=False,
                should_capture_stderr=False
            )
            ServerConstants.save_learning_process(process.pid)

        # start inference monitor server
        python_program = get_python_program()
        pip_source_dir = os.path.dirname(__file__)
        monitor_file = os.path.join(pip_source_dir, "device_model_monitor.py")
        self.monitor_process = ServerConstants.exec_console_with_shell_script_list(
            [
                python_program,
                monitor_file,
                "-v",
                self.args.version,
                "-ep",
                str(self.run_id),
                "-mi",
                str(model_id),
                "-mn",
                running_model_name,
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
        ServerConstants.save_learning_process(self.monitor_process.pid)
        self.mlops_metrics.broadcast_server_training_status(run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED)

        # forward deployment request to slave devices
        self.send_deployment_start_request_to_edges()

        self.client_mqtt_mgr.loop_forever()

    def reset_all_devices_status(self):
        edge_id_list = self.request_json["device_ids"]
        for edge_id in edge_id_list:
            self.mlops_metrics.report_client_training_status(edge_id, ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE)

    def stop_run(self):
        self.setup_client_mqtt_mgr()

        edge_id_list = self.request_json["device_ids"]
        self.send_deployment_stop_request_to_edges(edge_id_list, json.dumps(self.request_json))

        logging.info("Stop run successfully.")

        time.sleep(4)

        self.mlops_metrics.report_server_training_status(self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_KILLED)

        time.sleep(1)

        try:
            local_package_path = ServerConstants.get_package_download_dir()
            for package_file in listdir(local_package_path):
                if os.path.basename(package_file).startswith("run_" + str(self.run_id)):
                    shutil.rmtree(os.path.join(local_package_path, package_file), ignore_errors=True)
        except Exception as e:
            pass

    def stop_run_when_starting_failed(self):
        self.setup_client_mqtt_mgr()

        edge_id_list = self.request_json["device_ids"]
        logging.info("edge ids {}".format(str(edge_id_list)))
        self.send_exit_train_with_exception_request_to_edges(edge_id_list, json.dumps(self.request_json))

        logging.info("Stop run successfully when starting failed.")

        time.sleep(4)

        self.mlops_metrics.report_server_id_status(self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED)

        time.sleep(1)

    def cleanup_run_when_finished(self):
        if self.run_as_cloud_agent:
            self.stop_cloud_server()

        self.setup_client_mqtt_mgr()

        logging.info("Cleanup run successfully when finished.")

        self.mlops_metrics.broadcast_server_training_status(
            self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED
        )

        try:
            self.mlops_metrics.set_sys_reporting_status(False)
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
        if self.run_as_cloud_agent:
            self.stop_cloud_server()

        self.setup_client_mqtt_mgr()

        logging.info("Cleanup run successfully when starting failed.")

        self.mlops_metrics.broadcast_server_training_status(self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED)

        try:
            self.mlops_metrics.set_sys_reporting_status(False)
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

    def callback_deployment_result_message(self, topic=None, payload=None):
        # Save deployment result to local cache
        topic_splits = str(topic).split('/')
        device_id = topic_splits[-1]
        payload_json = json.loads(payload)
        end_point_id = payload_json["end_point_id"]
        model_id = payload_json["model_id"]
        model_name = payload_json["model_name"]
        model_version = payload_json["model_version"]
        model_status = payload_json["model_status"]
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            set_deployment_result(end_point_id, device_id, payload_json)
        self.slave_deployment_results_mapping[device_id] = model_status

        logging.info("callback_deployment_result_message: topic {}, payload {}, mapping {}.".format(
            topic, payload, self.slave_deployment_results_mapping))

        # When all deployments are finished
        device_id_list = self.request_json["device_ids"]
        if len(device_id_list) <= len(self.slave_deployment_results_mapping) + 1:
            is_exist_deployed_model = False
            failed_to_deploy_all_models = True
            for device_item in device_id_list:
                status = self.slave_deployment_results_mapping.\
                    get(str(device_item), ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED)
                if status == ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED:
                    pass
                else:
                    failed_to_deploy_all_models = False
                    if status == ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED:
                        is_exist_deployed_model = True
                        break

            # Failed to deploy models.
            if failed_to_deploy_all_models:
                # Send stage: MODEL_DEPLOYMENT_STAGE5 = "StartInferenceIngress"
                self.send_deployment_stages(self.run_id, model_name, model_id,
                                            "",
                                            ServerConstants.MODEL_DEPLOYMENT_STAGE5["index"],
                                            ServerConstants.MODEL_DEPLOYMENT_STAGE5["text"],
                                            "Failed to deploy the model to all devices.")
                FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                    set_deployment_result(end_point_id, self.edge_id, payload_json)
                self.release_client_mqtt_mgr()
                return
            if not is_exist_deployed_model:
                return

            # 1. We should generate one unified inference api
            ip = ServerConstants.get_local_ip()
            model_inference_port = ServerConstants.MODEL_INFERENCE_DEFAULT_PORT
            # model_inference_url = "http://{}:{}/api/v1/end_point_{}/model_id_{}" \
            #                       "/model_name_{}/model_version_{}/predict".format(ip, str(model_inference_port),
            #                                                                        end_point_id,
            #                                                                        model_id,
            #                                                                        model_name,
            #                                                                        model_version)
            if self.infer_host is not None and self.infer_host != "127.0.0.1" and self.infer_host != "localhost":
                ip = self.infer_host
            if ip.startswith("http://") or ip.startswith("https://"):
                model_inference_url = "{}/api/v1/predict".format(ip)
            else:
                model_inference_url = "http://{}:{}/api/v1/predict".format(ip, model_inference_port)

            # Send stage: MODEL_DEPLOYMENT_STAGE5 = "StartInferenceIngress"
            self.send_deployment_stages(self.run_id, model_name, model_id,
                                        model_inference_url,
                                        ServerConstants.MODEL_DEPLOYMENT_STAGE5["index"],
                                        ServerConstants.MODEL_DEPLOYMENT_STAGE5["text"],
                                        "inference url: {}".format(model_inference_url))

            # 2. We should send to MBE(ModelOps Backend)
            payload_json["model_url"] = model_inference_url
            payload_json["port"] = model_inference_port
            token = FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                get_end_point_token(end_point_id)
            payload_json["input_json"] = {"end_point_id": self.run_id,
                                          "model_id": model_id,
                                          "model_name": model_name,
                                          "model_version": model_version,
                                          "token": str(token),
                                          "data": "This is our test data. Please fill in here with your real data."}
            model_metadata = payload_json["model_metadata"]
            payload_json["output_json"] = {"outputs": model_metadata["outputs"]}
            FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                set_deployment_result(end_point_id, self.edge_id, payload_json)
            FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                set_end_point_activation(end_point_id, True)
            self.send_deployment_results_with_payload(self.run_id, payload_json)
            self.release_client_mqtt_mgr()

    def callback_deployment_status_message(self, topic=None, payload=None):
        # Save deployment status to local cache
        topic_splits = str(topic).split('/')
        device_id = topic_splits[-1]
        payload_json = json.loads(payload)
        end_point_id = payload_json["end_point_id"]
        model_status = payload_json["model_status"]
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port).set_deployment_status(end_point_id, device_id,
                                                                                             payload_json)
        self.slave_deployment_statuses_mapping[device_id] = model_status
        logging.info("callback_deployment_status_message: topic {}, payload {}, mapping {}.".format(
            topic, payload, self.slave_deployment_statuses_mapping))

        # When all deployments are finished
        device_id_list = self.request_json["device_ids"]
        if len(device_id_list) <= len(self.slave_deployment_statuses_mapping) + 1:
            is_exist_deployed_model = False
            failed_to_deploy_all_models = True
            for device_item in device_id_list:
                status = self.slave_deployment_statuses_mapping.\
                    get(str(device_item), ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED)
                if status == ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED:
                    pass
                else:
                    failed_to_deploy_all_models = False
                    if status == ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED:
                        is_exist_deployed_model = True
                        break

            # Failed to deploy the model to all devices
            if failed_to_deploy_all_models:
                FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                    set_end_point_activation(end_point_id, False)
                FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                    set_end_point_status(end_point_id, ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED)
                self.send_deployment_status(self.run_id, payload_json["model_name"], "",
                                            ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED)
                return
            if not is_exist_deployed_model:
                return

            # Send deployment finished message to ModelOps
            ip = ServerConstants.get_local_ip()
            model_inference_port = ServerConstants.MODEL_INFERENCE_DEFAULT_PORT
            if self.infer_host is not None and self.infer_host != "127.0.0.1" and self.infer_host != "localhost":
                ip = self.infer_host
            if ip.startswith("http://") or ip.startswith("https://"):
                model_inference_url = "{}/api/v1/predict".format(ip)
            else:
                model_inference_url = "http://{}:{}/api/v1/predict".format(ip, model_inference_port)
            FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                set_end_point_activation(end_point_id, True)
            FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                set_end_point_status(end_point_id, ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED)
            self.send_deployment_status(self.run_id, payload_json["model_name"],
                                        model_inference_url,
                                        ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED)

    def send_deployment_start_request_to_edges(self):
        run_id = self.request_json["run_id"]
        edge_id_list = self.request_json["device_ids"]
        logging.info("Edge ids: " + str(edge_id_list))
        for edge_id in edge_id_list:
            if edge_id == self.edge_id:
                continue
            # send start deployment request to each model device
            topic_start_deployment = "/model_ops/model_device/start_deployment/{}".format(str(edge_id))
            logging.info("start_deployment: send topic " + topic_start_deployment + " to client...")
            self.client_mqtt_mgr.send_message_json(topic_start_deployment, json.dumps(self.request_json))

    def send_deployment_delete_request_to_edges(self, payload, model_msg_object):
        edge_id_list = model_msg_object.device_ids
        logging.info("Device ids: " + str(edge_id_list))
        for edge_id in edge_id_list:
            if edge_id == self.edge_id:
                continue
            # send delete deployment request to each model device
            topic_delete_deployment = "/model_ops/model_device/delete_deployment/{}".format(str(edge_id))
            logging.info("delete_deployment: send topic " + topic_delete_deployment + " to client...")
            self.client_mqtt_mgr.send_message_json(topic_delete_deployment, payload)

    def callback_client_status_msg(self, topic=None, payload=None):
        payload_json = json.loads(payload)
        run_id = payload_json["run_id"]
        edge_id = payload_json["edge_id"]
        status = payload_json["status"]
        edge_id_list = self.request_json["device_ids"]
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

    def callback_start_deployment(self, topic, payload):
        """
        topic: /model_ops/model_device/start_deployment/model-agent-device-id
        payload: {"timestamp": 1671440005119, "end_point_id": 4325, "token": "FCpWU", "state": "STARTING","user_id": "105", "user_name": "alex.liang2", "device_ids": [693], "device_objs": [{"device_id": "0xT3630FW2YM@MacOS.Edge.Device", "os_type": "MacOS", "id": 693, "ip": "1.1.1.1", "memory": 1024, "cpu": "1.7", "gpu": "Nvidia", "extra_infos":{}}], "model_config": {"model_name": "image-model", "model_id": 111, "model_version": "v1", 'is_from_open": 0, "model_storage_url": "https://fedml.s3.us-west-1.amazonaws.com/1666239314792client-package.zip", "instance_scale_min": 1, "instance_scale_max": 3, "inference_engine": "onnx"}, "parameters": {"hidden_size": 128, "hidden_act": "gelu", "initializer_range": 0.02, "vocab_size": 30522, "hidden_dropout_prob": 0.1, "num_attention_heads": 2, "type_vocab_size": 2, "max_position_embeddings": 512, "num_hidden_layers": 2, "intermediate_size": 512, "attention_probs_dropout_prob": 0.1}}
        """
        logging.info("callback_start_deployment {}".format(payload))
        # get training params
        if self.run_as_cloud_server:
            message_bytes = payload.encode("ascii")
            base64_bytes = base64.b64decode(message_bytes)
            payload = base64_bytes.decode("ascii")
            logging.info("decoded payload: {}".format(payload))

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
        model_id = model_config["model_id"]
        model_storage_url = model_config["model_storage_url"]
        scale_min = model_config["instance_scale_min"]
        scale_max = model_config["instance_scale_max"]
        inference_engine = model_config["inference_engine"]
        inference_end_point_id = run_id

        run_id = inference_end_point_id
        self.args.run_id = run_id
        self.run_id = run_id
        request_json["run_id"] = run_id
        self.request_json = request_json
        self.running_request_json[str(run_id)] = request_json

        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            set_end_point_device_info(run_id, json.dumps(device_objs))
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            set_end_point_token(run_id, token)

        # Send stage: MODEL_DEPLOYMENT_STAGE1 = "Received"
        time.sleep(2)
        self.send_deployment_stages(self.run_id, model_name, model_id,
                                    "",
                                    ServerConstants.MODEL_DEPLOYMENT_STAGE1["index"],
                                    ServerConstants.MODEL_DEPLOYMENT_STAGE1["text"],
                                    "Received request for end point {}".format(run_id))
        time.sleep(1)

        # Send stage: MODEL_DEPLOYMENT_STAGE2 = "Initializing"
        self.send_deployment_stages(self.run_id, model_name, model_id,
                                    "",
                                    ServerConstants.MODEL_DEPLOYMENT_STAGE2["index"],
                                    ServerConstants.MODEL_DEPLOYMENT_STAGE2["text"],
                                    ServerConstants.MODEL_DEPLOYMENT_STAGE2["text"])

        ServerConstants.save_runner_infos(self.args.device_id + "." + self.args.os_name, self.edge_id, run_id=run_id)
        time.sleep(1)

        # Start server with multi-processing mode
        if not self.run_as_cloud_server:
            # Setup MQTT message listener to the client status message from the server.
            edge_id_list = request_json["device_ids"]
            for edge_id in edge_id_list:
                topic_name = "fl_client/flclient_agent_" + str(edge_id) + "/status"
                self.mqtt_mgr.add_message_listener(topic_name, self.callback_client_status_msg)
                self.mqtt_mgr.subscribe_msg(topic_name)

        if self.run_as_edge_server_and_agent:
            # Start log processor for current run
            MLOpsRuntimeLogDaemon.get_instance(self.args).set_log_source(
                ServerConstants.FEDML_LOG_SOURCE_TYPE_MODEL_END_POINT)
            MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(run_id, self.edge_id)
            self.args.run_id = run_id

            server_runner = FedMLServerRunner(
                self.args, run_id=run_id, request_json=request_json, agent_config=self.agent_config
            )
            server_runner.run_as_edge_server_and_agent = self.run_as_edge_server_and_agent
            server_runner.edge_id = self.edge_id
            server_runner.infer_host = self.infer_host
            server_runner.redis_addr = self.redis_addr
            server_runner.redis_port = self.redis_port
            server_runner.redis_password = self.redis_password
            # server_runner.run()
            server_process = Process(target=server_runner.run)
            server_process.start()
            ServerConstants.save_run_process(server_process.pid)

            # Send stage: MODEL_DEPLOYMENT_STAGE3 = "StartRunner"
            self.send_deployment_stages(self.run_id, model_name, model_id,
                                        "",
                                        ServerConstants.MODEL_DEPLOYMENT_STAGE3["index"],
                                        ServerConstants.MODEL_DEPLOYMENT_STAGE3["text"],
                                        ServerConstants.MODEL_DEPLOYMENT_STAGE3["text"])

        elif self.run_as_cloud_agent:
            # Start log processor for current run
            MLOpsRuntimeLogDaemon.get_instance(self.args).set_log_source(
                ServerConstants.FEDML_LOG_SOURCE_TYPE_MODEL_END_POINT)
            MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(
                run_id, self.request_json.get("cloudServerDeviceId", "0")
            )

            server_runner = FedMLServerRunner(
                self.args, run_id=run_id, request_json=request_json, agent_config=self.agent_config
            )
            server_runner.run_as_cloud_agent = self.run_as_cloud_agent
            server_runner.infer_host = self.infer_host
            server_runner.redis_addr = self.redis_addr
            server_runner.redis_port = self.redis_port
            server_runner.redis_password = self.redis_password
            server_process = Process(target=server_runner.start_cloud_server_process)
            server_process.start()
            ServerConstants.save_run_process(server_process.pid)

            # Send stage: MODEL_DEPLOYMENT_STAGE3 = "StartRunner"
            self.send_deployment_stages(self.run_id, model_name, model_id,
                                        "",
                                        ServerConstants.MODEL_DEPLOYMENT_STAGE3["index"],
                                        ServerConstants.MODEL_DEPLOYMENT_STAGE3["text"],
                                        ServerConstants.MODEL_DEPLOYMENT_STAGE3["text"])

        elif self.run_as_cloud_server:
            self.server_agent_id = self.request_json.get("cloud_agent_id", self.edge_id)
            run_id = self.request_json["run_id"]

            # Start log processor for current run
            self.args.run_id = run_id
            MLOpsRuntimeLogDaemon.get_instance(self.args).set_log_source(
                ServerConstants.FEDML_LOG_SOURCE_TYPE_MODEL_END_POINT)
            MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(run_id, self.edge_id)
            self.run()

    def callback_stop_deployment(self, topic, payload):
        logging.info("callback_stop_deployment: topic = %s, payload = %s" % (topic, payload))

        request_json = json.loads(payload)
        run_id = request_json["run_id"]
        edge_id_list = request_json["device_ids"]
        server_id = request_json["serverId"]

        logging.info("Stop deployment with multiprocessing...")

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

    def callback_activate_deployment(self, topic, payload):
        logging.info("callback_activate_deployment: topic = %s, payload = %s" % (topic, payload))

        # Parse payload as the model message object.
        model_msg_object = FedMLModelMsgObject(topic, payload)

        # If the previous deployment did not complete successfully, we need to restart the deployment.
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        prev_status = FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            get_end_point_status(model_msg_object.inference_end_point_id)
        if prev_status != ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED:
            prev_deployment_result = FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                get_idle_device(model_msg_object.inference_end_point_id, model_msg_object.model_id,
                                check_end_point_status=False)
            if prev_deployment_result is None:
                self.callback_start_deployment(topic, payload)
                return

        # Set end point as activated status
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            set_end_point_activation(model_msg_object.inference_end_point_id, True)

        # Send deployment status to the ModelOps backend
        if prev_status == ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED:
            self.send_deployment_status(model_msg_object.inference_end_point_id,
                                        model_msg_object.model_name, "", prev_status)

    def callback_deactivate_deployment(self, topic, payload):
        logging.info("callback_deactivate_deployment: topic = %s, payload = %s" % (topic, payload))

        # Parse payload as the model message object.
        model_msg_object = FedMLModelMsgObject(topic, payload)

        # Set end point as deactivated status
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            set_end_point_activation(model_msg_object.inference_end_point_id, False)

    def callback_delete_deployment(self, topic, payload):
        logging.info("callback_delete_deployment: topic = %s, payload = %s" % (topic, payload))

        # Parse payload as the model message object.
        model_msg_object = FedMLModelMsgObject(topic, payload)

        # Set end point as deactivated status
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            set_end_point_activation(model_msg_object.inference_end_point_id, False)

        self.setup_client_mqtt_mgr()
        self.send_deployment_delete_request_to_edges(payload, model_msg_object)

    def send_deployment_results_with_payload(self, end_point_id, payload):
        self.send_deployment_results(end_point_id, payload["model_name"], payload["model_url"],
                                     payload["model_version"], payload["port"],
                                     payload["inference_engine"],
                                     payload["model_metadata"],
                                     payload["model_config"],
                                     payload["input_json"],
                                     payload["output_json"])

    def send_deployment_results(self, end_point_id, model_name, model_inference_url,
                                model_version, inference_port, inference_engine,
                                model_metadata, model_config, input_json, output_json):
        deployment_results_topic_prefix = "/model_ops/model_device/return_deployment_result"
        deployment_results_topic = "{}/{}".format(deployment_results_topic_prefix, end_point_id)
        deployment_results_payload = {"end_point_id": end_point_id,
                                      "model_name": model_name, "model_url": model_inference_url,
                                      "version": model_version, "port": inference_port,
                                      "inference_engine": inference_engine,
                                      "model_metadata": model_metadata,
                                      "model_config": model_config,
                                      "input_json": input_json,
                                      "output_json": output_json,
                                      "timestamp": int(format(time.time_ns()/1000.0, '.0f'))}
        self.setup_client_mqtt_mgr()
        self.client_mqtt_mgr.send_message_json(deployment_results_topic, json.dumps(deployment_results_payload))
        self.client_mqtt_mgr.send_message_json(deployment_results_topic_prefix, json.dumps(deployment_results_payload))

    def send_deployment_status(self, end_point_id, model_name, model_inference_url, model_status):
        deployment_status_topic_prefix = "/model_ops/model_device/return_deployment_status"
        deployment_status_topic = "{}/{}".format(deployment_status_topic_prefix, end_point_id)
        deployment_status_payload = {"end_point_id": end_point_id, "model_name": model_name,
                                     "model_url": model_inference_url,
                                     "model_status": model_status,
                                     "timestamp": int(format(time.time_ns()/1000.0, '.0f'))}
        self.setup_client_mqtt_mgr()
        self.client_mqtt_mgr.send_message_json(deployment_status_topic, json.dumps(deployment_status_payload))
        self.client_mqtt_mgr.send_message_json(deployment_status_topic_prefix, json.dumps(deployment_status_payload))

    def send_deployment_stages(self, end_point_id, model_name, model_id, model_inference_url,
                               model_stages_index, model_stages_title, model_stage_detail):
        deployment_stages_topic_prefix = "/model_ops/model_device/return_deployment_stages"
        deployment_stages_topic = "{}/{}".format(deployment_stages_topic_prefix, end_point_id)
        deployment_stages_payload = {"model_name": model_name,
                                     "model_id": model_id,
                                     "model_url": model_inference_url,
                                     "end_point_id": end_point_id,
                                     "model_stage_index": model_stages_index,
                                     "model_stage_title": model_stages_title,
                                     "model_stage_detail": model_stage_detail,
                                     "timestamp": int(format(time.time_ns()/1000.0, '.0f'))}
        logging.info("-----Stages{}:{}-----".format(model_stages_index, model_stages_title))
        logging.info("-----Stages{}:{}.....".format(model_stages_index, model_stage_detail))
        self.setup_client_mqtt_mgr()
        self.client_mqtt_mgr.send_message_json(deployment_stages_topic, json.dumps(deployment_stages_payload))
        self.client_mqtt_mgr.send_message_json(deployment_stages_topic_prefix, json.dumps(deployment_stages_payload))

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

        logging.info("on_client_mqtt_connected: {}.".format(self.client_mqtt_is_connected))

        self.subscribe_slave_devices_message()

    def setup_client_mqtt_mgr(self):
        if self.client_mqtt_mgr is not None:
            return

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
            "FedML_ServerAgent_Metrics_{}_{}".format(self.args.current_device_id, str(os.getpid())),
        )

        if self.mlops_metrics is None:
            self.mlops_metrics = MLOpsMetrics()
        self.mlops_metrics.set_messenger(self.client_mqtt_mgr)
        self.mlops_metrics.run_id = self.run_id
        self.mlops_metrics.edge_id = self.edge_id
        self.mlops_metrics.server_agent_id = self.server_agent_id

        self.client_mqtt_mgr.add_connected_listener(self.on_client_mqtt_connected)
        self.client_mqtt_mgr.add_disconnected_listener(self.on_client_mqtt_disconnected)
        self.client_mqtt_mgr.connect()

    def release_client_mqtt_mgr(self):
        time.sleep(1)
        self.client_mqtt_mgr.loop_stop()
        self.client_mqtt_mgr.disconnect()

    def send_deployment_stop_request_to_edges(self, edge_id_list, payload):
        for edge_id in edge_id_list:
            topic_stop_deployment = "/model_ops/model_device/stop_deployment/{}".format(str(self.edge_id))
            logging.info("stop_deployment: send topic " + topic_stop_deployment)
            self.client_mqtt_mgr.send_message_json(topic_stop_deployment, payload)

    def send_exit_train_with_exception_request_to_edges(self, edge_id_list, payload):
        for edge_id in edge_id_list:
            topic_exit_train = "flserver_agent/" + str(edge_id) + "/exit_train_with_exception"
            logging.info("exit_train_with_exception: send topic " + topic_exit_train)
            self.client_mqtt_mgr.send_message_json(topic_exit_train, payload)

    def callback_runner_id_status(self, topic, payload):
        logging.info("callback_runner_id_status: topic = %s, payload = %s" % (topic, payload))

        request_json = json.loads(payload)
        run_id = request_json["run_id"]
        status = request_json["status"]
        edge_id = request_json["edge_id"]

        if (
                status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED
                or status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED
        ):
            logging.info("Received training finished message.")

            logging.info("Will end training server.")

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
        role = ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_ON_PREMISE_MASTER_INDEX]
        if self.run_as_edge_server_and_agent:
            role = ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_ON_PREMISE_MASTER_INDEX]
        elif self.run_as_cloud_agent:
            role = ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_FEDML_CLOUD_MASTER_INDEX]
        elif self.run_as_cloud_server:
            role = ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_INFERENCE_INSTANCE_INDEX]

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

    def subscribe_slave_devices_message(self):
        run_id = self.request_json["run_id"]
        edge_id_list = self.request_json["device_ids"]
        logging.info("Edge ids: " + str(edge_id_list))
        for edge_id in edge_id_list:
            if str(edge_id) == str(self.edge_id):
                continue
            # subscribe deployment result message for each model device
            deployment_results_topic = "/model_ops/model_device/return_deployment_result/{}".format(edge_id)
            self.client_mqtt_mgr.add_message_listener(deployment_results_topic, self.callback_deployment_result_message)
            self.client_mqtt_mgr.subscribe_msg(deployment_results_topic)

            # subscribe deployment status message for each model device
            deployment_status_topic = "/model_ops/model_device/return_deployment_status/{}".format(edge_id)
            self.client_mqtt_mgr.add_message_listener(deployment_status_topic, self.callback_deployment_status_message)
            self.client_mqtt_mgr.subscribe_msg(deployment_status_topic)

            logging.info("subscribe device messages {}, {}".format(
                deployment_results_topic, deployment_status_topic))

    def on_agent_mqtt_connected(self, mqtt_client_object):
        # The MQTT message topic format is as follows: <sender>/<receiver>/<action>

        # Setup MQTT message listener for starting deployment
        server_agent_id = self.edge_id
        topic_start_deployment = "/model_ops/model_device/start_deployment/{}".format(str(self.edge_id))
        self.mqtt_mgr.add_message_listener(topic_start_deployment, self.callback_start_deployment)

        # Setup MQTT message listener for stopping deployment
        topic_stop_deployment = "/model_ops/model_device/stop_deployment/{}".format(str(self.edge_id))
        self.mqtt_mgr.add_message_listener(topic_stop_deployment, self.callback_stop_deployment)

        # Setup MQTT message listener for activating deployment
        topic_activate_deployment = "/model_ops/model_device/activate_deployment/{}".format(str(self.edge_id))
        self.mqtt_mgr.add_message_listener(topic_activate_deployment, self.callback_activate_deployment)

        # Setup MQTT message listener for deactivating deployment
        topic_deactivate_deployment = "/model_ops/model_device/deactivate_deployment/{}".format(str(self.edge_id))
        self.mqtt_mgr.add_message_listener(topic_deactivate_deployment, self.callback_deactivate_deployment)

        # Setup MQTT message listener for delete deployment
        topic_delete_deployment = "/model_ops/model_device/delete_deployment/{}".format(str(self.edge_id))
        self.mqtt_mgr.add_message_listener(topic_delete_deployment, self.callback_delete_deployment)

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
        mqtt_client_object.subscribe(topic_start_deployment, qos=2)
        mqtt_client_object.subscribe(topic_stop_deployment, qos=2)
        mqtt_client_object.subscribe(topic_activate_deployment, qos=2)
        mqtt_client_object.subscribe(topic_deactivate_deployment, qos=2)
        mqtt_client_object.subscribe(topic_delete_deployment, qos=2)
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
        click.echo("Congratulations, you have logged into the FedML ModelOps platform successfully!")
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

        MLOpsRuntimeLogDaemon.get_instance(self.args).stop_all_log_processor()

        # Setup MQTT connected listener
        self.mqtt_mgr.add_connected_listener(self.on_agent_mqtt_connected)
        self.mqtt_mgr.add_disconnected_listener(self.on_agent_mqtt_disconnected)
        self.mqtt_mgr.connect()

        self.setup_client_mqtt_mgr()
        self.mlops_metrics.report_server_training_status(self.run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE)
        MLOpsStatus.get_instance().set_server_agent_status(
            self.edge_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE
        )
        self.mlops_metrics.set_sys_reporting_status(enable=True, is_client=False)
        setattr(self.args, "mqtt_config_path", service_config["mqtt_config"])
        self.mlops_metrics.report_sys_perf(self.args, is_client=False)

    def start_agent_mqtt_loop(self):
        # Start MQTT message loop
        try:
            self.mqtt_mgr.loop_forever()
        except Exception as e:
            pass
