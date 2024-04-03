import copy
import json
import logging
import os
import time
import queue
import traceback
from abc import ABC
from multiprocessing import Queue

import fedml
from fedml.core.mlops import MLOpsRuntimeLog
from .device_client_constants import ClientConstants
from .device_model_cache import FedMLModelCache
from .device_server_constants import ServerConstants
from .device_server_data_interface import FedMLServerDataInterface
from ..comm_utils import sys_utils
from ..comm_utils.run_process_utils import RunProcessUtils
from ..comm_utils.sys_utils import get_python_program
from ..scheduler_core.general_constants import GeneralConstants
from ..master.base_master_job_runner import FedMLBaseMasterJobRunner
from .device_replica_controller import FedMLDeviceReplicaController
from .job_runner_msg_sender import FedMLDeployJobRunnerMsgSender


class FedMLDeployMasterJobRunner(FedMLBaseMasterJobRunner, FedMLDeployJobRunnerMsgSender, ABC):

    default_redis_addr = "local"
    default_redis_port = "6379"
    default_redis_password = "fedml_default"

    def __init__(self, args, run_id=0, request_json=None, agent_config=None, edge_id=0,
                 cuda_visible_gpu_ids_str=None):
        FedMLDeployJobRunnerMsgSender.__init__(self)
        FedMLBaseMasterJobRunner.__init__(
            self, args, edge_id=edge_id, request_json=request_json, agent_config=agent_config, run_id=run_id,
            cuda_visible_gpu_ids_str=cuda_visible_gpu_ids_str, agent_data_dir=ServerConstants.get_data_dir(),
            agent_package_download_dir=ServerConstants.get_package_download_dir(),
            agent_package_unzip_dir=GeneralConstants.get_package_unzip_dir(ServerConstants.get_package_download_dir()),
            agent_log_file_dir=ServerConstants.get_log_file_dir()
        )

        self.infer_host = "127.0.0.1"
        self.redis_addr = "local"
        self.redis_port = "6379"
        self.redis_password = "fedml_default"
        self.inference_gateway_process = None
        self.monitor_process = None
        self.replica_controller = None
        self.deployed_replica_payload = None
        self.slave_deployment_results_map = dict()
        self.deployment_result_queue = Queue()

    # Override
    def _generate_job_runner_instance(self, args, run_id=None, request_json=None, agent_config=None, edge_id=None,):
        return FedMLDeployMasterJobRunner(
            args, run_id=run_id, request_json=request_json, agent_config=self.agent_config, edge_id=edge_id
        )

    # Override
    def _generate_extend_queue_list(self):
        return [self.deployment_result_queue]

    # Override
    def run_impl(
        self, edge_id_status_queue, edge_device_info_queue, run_metrics_queue,
        run_event_queue, run_artifacts_queue, run_logs_queue, edge_device_info_global_queue,
        run_extend_queue_list=None, sender_message_queue=None, listener_message_queue=None,
        status_center_queue=None
    ):
        # Parse the model parameters.
        run_id, end_point_name, token, user_id, user_name, device_ids, device_objs, model_config, model_name, \
            model_id, model_storage_url, scale_min, scale_max, inference_engine, model_is_from_open, \
            inference_end_point_id, use_gpu, memory_size, model_version, inference_port = \
            FedMLDeployMasterJobRunner.parse_model_run_params(self.request_json)

        # Print request parameters.
        logging.info("model deployment request: {}".format(self.request_json))
        logging.info("send deployment stages...")

        # Generate the replica controller object.
        self.replica_controller = FedMLDeviceReplicaController(self.edge_id, self.request_json)

        # Start the process to report system performance(cpu,memory,etc.) to MLOps
        self.mlops_metrics.report_sys_perf(self.args, self.agent_config["mqtt_config"], run_id=run_id)

        # Check if we should stop the runner
        self.check_runner_stop_event()

        # Send stage: MODEL_DEPLOYMENT_STAGE4 = "ForwardRequest2Slave"
        self.send_deployment_stages(
            self.run_id, model_name, model_id, "", ServerConstants.MODEL_DEPLOYMENT_STAGE4["index"],
            ServerConstants.MODEL_DEPLOYMENT_STAGE4["text"], ServerConstants.MODEL_DEPLOYMENT_STAGE4["text"],
            message_center=self.message_center)

        # Init the runtime logs
        self.args.run_id = self.run_id
        MLOpsRuntimeLog.get_instance(self.args).init_logs(log_level=logging.INFO)

        # Report server running status
        logging.info("report deployment status...")
        self.check_runner_stop_event()
        self.status_reporter.report_server_id_status(
            run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_STARTING,
            is_from_model=True, running_json=json.dumps(self.request_json),
            server_agent_id=self.edge_id, server_id=self.edge_id, edge_id=self.edge_id)
        self.send_deployment_status(
            self.run_id, end_point_name, model_name, "",
            ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYING,
            message_center=self.message_center)

        # start unified inference server
        self.start_device_inference_gateway(
            run_id, end_point_name, model_id, model_name, model_version,
            agent_config=self.agent_config, inference_port=inference_port)

        # start inference monitor server
        self.stop_device_inference_monitor(
            run_id, end_point_name, model_id, model_name, model_version)
        self.start_device_inference_monitor(
            run_id, end_point_name, model_id, model_name, model_version,
            redis_addr=self.redis_addr, redis_port=self.redis_port, redis_password=self.redis_password
        )

        # Changed the status to "IDLE"
        self.status_reporter.report_server_id_status(
            run_id, ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED,
            is_from_model=True, server_agent_id=self.edge_id, server_id=self.edge_id, edge_id=self.edge_id,)

        # Check if we should stop the runner
        logging.info("send the model inference request to slave devices...")
        self.check_runner_stop_event()

        # Forward deployment request to slave devices
        # Handle "op:add" && "op:remove"
        devices_sent_add_or_remove_msg = self.send_deployment_start_request_to_edges()

        # Handle "op:update"
        devices_sent_update_remove_msg = self.send_first_scroll_update_msg()

        if len(devices_sent_add_or_remove_msg) == 0 and len(devices_sent_update_remove_msg) == 0:
            # No device is added or removed, and no device is updated or removed
            ip = GeneralConstants.get_ip_address(self.request_json)
            master_port = os.getenv("FEDML_MASTER_PORT", None)
            if master_port is not None:
                inference_port = int(master_port)
            model_inference_port = inference_port
            if ip.startswith("http://") or ip.startswith("https://"):
                model_inference_url = "{}/api/v1/predict".format(ip)
            else:
                model_inference_url = "http://{}:{}/api/v1/predict".format(ip, model_inference_port)

            self.send_deployment_status(
                run_id, end_point_name, model_name, model_inference_url,
                ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED,
                message_center=self.message_center
            )

            self.trigger_completed_event()
            return

        self.deployment_result_queue = run_extend_queue_list[0]
        while True:
            self.check_runner_stop_event()

            try:
                deployment_result = self.deployment_result_queue.get(block=False, timeout=0.2)
                result_topic = deployment_result.get("topic", None)
                result_payload = deployment_result.get("payload", None)
                self.process_deployment_result_message(topic=result_topic, payload=result_payload)
            except queue.Empty as e:  # If queue is empty, then continue
                pass

            time.sleep(0.5)

    def save_deployment_result(self, topic=None, payload=None):
        self.deployment_result_queue.put({"topic": topic, "payload": payload})

    def process_deployment_result_message(self, topic=None, payload=None):
        # Parse the parameters
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

        # Save deployment result to local cache
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
            self.save_deployed_replica_payload(payload_json)
        else:
            if model_status != ClientConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED:
                logging.error(f"Unsupported model status {model_status}.")
            self.send_deployment_status(
                end_point_id, end_point_name, payload_json["model_name"], "",
                ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED,
                message_center=self.message_center
            )

        # Notify the replica number controller
        self.callback_update_curr_replica_num_state(device_id, replica_no, model_status)

        # Notify the replica version controller, which might trigger the next rolling update
        self.send_next_scroll_update_msg(device_id, replica_no)

        # Update the global deployment result mapping
        self.slave_deployment_results_map[str(device_id)] = model_status

        # Check if the endpoint is running
        request_json = self.request_json
        if request_json is None:
            logging.error(f"The endpoint {end_point_id} is not running.")
            self.send_deployment_status(
                end_point_id, end_point_name, payload_json["model_name"], "",
                ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED,
                message_center=self.message_center
            )
            return

        # Wait for all replica's result, not device-level
        if self.is_all_replica_num_reconciled() and self.is_all_replica_version_reconciled():
            '''
            When all the devices have finished the add / delete / update operation
            '''
            # 1. We should generate one unified inference api
            # Note that here we use the gateway port instead of the inference port that is used by the slave device
            model_config_parameters = request_json["parameters"]
            inference_port = model_config_parameters.get("server_internal_port",
                                                         ServerConstants.MODEL_INFERENCE_DEFAULT_PORT)
            inference_port_external = model_config_parameters.get("server_external_port", inference_port)
            ip = GeneralConstants.get_ip_address(request_json)

            if ip.startswith("http://") or ip.startswith("https://"):
                model_inference_url = "{}/inference/{}".format(ip, end_point_id)
            else:
                model_inference_url = "http://{}:{}/inference/{}".format(ip, inference_port_external, end_point_id)

            # Send stage: MODEL_DEPLOYMENT_STAGE5 = "StartInferenceIngress"
            self.send_deployment_stages(
                end_point_id, model_name, model_id, model_inference_url,
                ServerConstants.MODEL_DEPLOYMENT_STAGE5["index"], ServerConstants.MODEL_DEPLOYMENT_STAGE5["text"],
                "inference url: {}".format(model_inference_url), message_center=self.message_center)

            # Prepare the result to MLOps
            deployed_replica_payload = self.get_deployed_replica_payload()
            if deployed_replica_payload is not None:
                payload_json = deployed_replica_payload
                model_slave_url = payload_json["model_url"]
                payload_json["model_url"] = model_inference_url
                payload_json["port"] = inference_port_external
                token = FedMLModelCache.get_instance(self.redis_addr, self.redis_port).get_end_point_token(
                    end_point_id, end_point_name, model_name)

                model_metadata = payload_json["model_metadata"]
                model_inputs = model_metadata["inputs"]
                ret_inputs = list()
                if "type" in model_metadata and model_metadata["type"] == "default":
                    payload_json["input_json"] = {
                        "end_point_name": end_point_name, "model_name": model_name, "token": str(token),
                        "inputs": model_inputs, "outputs": []}
                    payload_json["output_json"] = model_metadata["outputs"]
                else:
                    raise Exception(f"Unsupported model metadata type {model_metadata['type']}")

                self.send_deployment_results_with_payload(
                    end_point_id, end_point_name, payload_json)

                payload_json_saved = payload_json
                payload_json_saved["model_slave_url"] = model_slave_url
                FedMLServerDataInterface.get_instance().save_job_result(end_point_id, self.edge_id,
                                                                        json.dumps(payload_json_saved))
            else:
                # Arrive here because only contains remove ops, so we do not need to update the model metadata
                pass

            FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                set_end_point_activation(end_point_id, end_point_name, True)

            self.send_deployment_status(
                end_point_id, end_point_name, payload_json["model_name"],
                model_inference_url, ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED,
                message_center=self.message_center
            )

            self.trigger_completed_event()

    @staticmethod
    def start_device_inference_gateway(
            run_id, end_point_name, model_id,
            model_name, model_version, inference_port=ServerConstants.MODEL_INFERENCE_DEFAULT_PORT,
            agent_config=None, redis_addr=None, redis_port=None, redis_password=None
    ):
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
                    agent_config["mqtt_config"]["BROKER_HOST"] + connect_str +
                    str(agent_config["mqtt_config"]["BROKER_PORT"]) + connect_str +
                    agent_config["mqtt_config"]["MQTT_USER"] + connect_str +
                    agent_config["mqtt_config"]["MQTT_PWD"] + connect_str +
                    str(agent_config["mqtt_config"]["MQTT_KEEPALIVE"]), "FEDML@9999GREAT")
                inference_gateway_process = ServerConstants.exec_console_with_script(
                    "REDIS_ADDR=\"{}\" REDIS_PORT=\"{}\" REDIS_PASSWORD=\"{}\" "
                    "END_POINT_NAME=\"{}\" "
                    "MODEL_NAME=\"{}\" MODEL_VERSION=\"{}\" MODEL_INFER_URL=\"{}\" VERSION=\"{}\" "
                    "USE_MQTT_INFERENCE={} USE_WORKER_GATEWAY={} EXT_INFO={} "
                    "{} -m uvicorn {} --host 0.0.0.0 --port {} --reload --reload-delay 3 --reload-dir {} "
                    "--log-level critical".format(
                        redis_addr, redis_port, redis_password, end_point_name,
                        model_name, model_version, "", fedml.get_env_version(), use_mqtt_inference,
                        use_worker_gateway, ext_info, python_program, inference_gw_cmd, str(inference_port),
                        fedml_base_dir),
                    should_capture_stdout=False, should_capture_stderr=False)

                return inference_gateway_process

        return None

    @staticmethod
    def start_device_inference_monitor(
            run_id, end_point_name, model_id, model_name, model_version, check_stopped_event=True,
            redis_addr=None, redis_port=None, redis_password=None
    ):
        # start inference monitor server
        logging.info(f"start the model inference monitor, end point {run_id}, model name {model_name}...")
        run_id_str = str(run_id)
        pip_source_dir = os.path.dirname(__file__)
        monitor_file = os.path.join(pip_source_dir, "device_model_monitor.py")
        python_program = get_python_program()
        running_model_name = ServerConstants.get_running_model_name(end_point_name,
                                                                    model_name, model_version, run_id, model_id)
        monitor_process = ServerConstants.exec_console_with_shell_script_list(
            [python_program, monitor_file, "-v", fedml.get_env_version(), "-ep", run_id_str,
             "-epn", str(end_point_name), "-mi", str(model_id), "-mn", model_name,
             "-mv", model_version, "-iu", "infer_url", "-ra", redis_addr,
             "-rp", redis_port, "-rpw", redis_password],
            should_capture_stdout=False, should_capture_stderr=False
        )
        return monitor_process

    @staticmethod
    def stop_device_inference_monitor(run_id, end_point_name, model_id, model_name, model_version):
        # stop inference monitor server
        logging.info(f"stop the model inference monitor, end point {run_id}, model name {model_name}...")
        sys_utils.cleanup_model_monitor_processes(run_id, end_point_name,
                                                  model_id, model_name, model_version)

    @staticmethod
    def recover_inference_and_monitor(redis_addr=None, redis_port=None, redis_password=None):
        # noinspection PyBroadException
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
                    FedMLDeployMasterJobRunner.parse_model_run_params(json.loads(job.running_json))

                FedMLModelCache.get_instance().set_redis_params(redis_addr, redis_password)
                is_activated = FedMLModelCache.get_instance(redis_addr, redis_port). \
                    get_end_point_activation(run_id)
                if not is_activated:
                    continue

                FedMLDeployMasterJobRunner.start_device_inference_gateway(
                    run_id, end_point_name, model_id, model_name, model_version, inference_port=inference_port)

                FedMLDeployMasterJobRunner.stop_device_inference_monitor(
                    run_id, end_point_name, model_id, model_name, model_version)
                FedMLDeployMasterJobRunner.start_device_inference_monitor(
                    run_id, end_point_name, model_id, model_name, model_version,
                    redis_addr=FedMLDeployMasterJobRunner.default_redis_addr,
                    redis_port=FedMLDeployMasterJobRunner.default_redis_port,
                    redis_password=FedMLDeployMasterJobRunner.default_redis_password
                )
        except Exception as e:
            logging.info("recover inference and monitor: {}".format(traceback.format_exc()))

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

    def send_next_scroll_update_msg(self, device_id, replica_no):
        if replica_no is None:
            return

        replica_controller = self.replica_controller

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

    def save_deployed_replica_payload(self, payload_json):
        self.deployed_replica_payload = copy.deepcopy(payload_json)

    def get_deployed_replica_payload(self):
        return self.deployed_replica_payload

    def callback_update_curr_replica_num_state(self, changed_device_id, replica_no, op_type):
        if self.replica_controller is not None:
            self.replica_controller.callback_update_curr_replica_num_state(changed_device_id, replica_no, op_type)

    def is_all_replica_num_reconciled(self):
        if self.replica_controller is not None:
            return self.replica_controller.is_all_replica_num_reconciled()

        return False

    def is_all_replica_version_reconciled(self):
        if self.replica_controller is not None:
            return self.replica_controller.is_all_replica_version_reconciled()

        return False

    @staticmethod
    def generate_request_json_with_replica_diff(run_id, edge_id, request_json):
        # Replica Controller is per deployment!
        replica_controller = FedMLDeviceReplicaController(edge_id, request_json)
        logging.info(f"Start Diff Replica controller for run {run_id} on edge {edge_id}")

        # Prepare num diff
        run_id_str = str(run_id)
        new_request_with_num_diff = replica_controller.generate_diff_to_request_json()
        request_json = new_request_with_num_diff

        # Prepare version diff
        new_request_with_version_diff = replica_controller.init_first_update_device_replica_mapping()
        request_json = new_request_with_version_diff

        return request_json

    @staticmethod
    def parse_model_run_params(running_json):
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

    # Override
    def get_download_package_info(self, packages_config=None):
        model_name = packages_config["model_name"]
        model_storage_url = packages_config["model_storage_url"]
        return model_name, model_storage_url

    # Override
    def build_dynamic_args(self, run_id, run_config, package_conf_object, base_dir):
        pass

    # Override
    def build_dynamic_constrain_variables(self, run_id, run_config):
        pass
