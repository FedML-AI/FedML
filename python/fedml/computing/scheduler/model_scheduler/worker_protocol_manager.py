
import json
import logging
import os
import traceback

from fedml.computing.scheduler.comm_utils.job_utils import JobRunnerUtils
from fedml.computing.scheduler.comm_utils.run_process_utils import RunProcessUtils
from fedml.computing.scheduler.comm_utils.sys_utils import get_python_program
from fedml.core.mlops import MLOpsConfigs, MLOpsRuntimeLog, MLOpsRuntimeLogDaemon
from .device_model_db import FedMLModelDatabase
from .device_model_msg_object import FedMLModelMsgObject
from .device_client_constants import ClientConstants
from .device_client_data_interface import FedMLClientDataInterface
from ..slave.base_slave_protocol_manager import FedMLBaseSlaveProtocolManager
from .worker_job_runner_manager import FedMLDeployJobRunnerManager
from .device_mqtt_inference_protocol import FedMLMqttInference
from ..scheduler_core.compute_cache_manager import ComputeCacheManager
from .device_model_cache import FedMLModelCache


class FedMLDeployWorkerProtocolManager(FedMLBaseSlaveProtocolManager):
    def __init__(self, args, agent_config=None):
        FedMLBaseSlaveProtocolManager.__init__(self, args, agent_config=agent_config)

        self.message_center_name = "deploy_slave_agent"
        self.is_deployment_status_center = True

        self.topic_start_deployment = None
        self.topic_delete_deployment = None

        self.infer_host = "127.0.0.1"
        self.redis_addr = "local"
        self.redis_port = "6379"
        self.redis_password = "fedml_default"
        self.endpoint_sync_protocol = None
        self.local_api_process = None
        self.mqtt_inference_obj = None

    # Override
    def _generate_protocol_manager_instance(self, args, agent_config=None):
        return FedMLDeployWorkerProtocolManager(args, agent_config=agent_config)

    # Override
    def generate_topics(self):
        super().generate_topics()

        # The topic for start deployment
        self.topic_start_deployment = "model_ops/model_device/start_deployment/{}".format(str(self.edge_id))

        # The topic for deleting endpoint
        self.topic_delete_deployment = "model_ops/model_device/delete_deployment/{}".format(str(self.edge_id))

        # Subscribe topics for endpoints
        self.add_subscribe_topic(self.topic_start_deployment)
        self.add_subscribe_topic(self.topic_delete_deployment)

    # Override
    def add_protocol_handler(self):
        super().add_protocol_handler()

        # Add the message listeners for endpoint related topics
        self.add_message_listener(self.topic_start_deployment, self.callback_start_deployment)
        self.add_message_listener(self.topic_delete_deployment, self.callback_delete_deployment)

    # Override
    def _get_job_runner_manager(self):
        return FedMLDeployJobRunnerManager.get_instance()

    # Override
    def _init_extra_items(self):
        # Init local database
        FedMLClientDataInterface.get_instance().create_job_table()
        try:
            FedMLModelDatabase.get_instance().set_database_base_dir(ClientConstants.get_database_dir())
            FedMLModelDatabase.get_instance().create_table()
        except Exception as e:
            pass

        client_api_cmd = "fedml.computing.scheduler.model_scheduler.device_client_api:api"
        client_api_pids = RunProcessUtils.get_pid_from_cmd_line(client_api_cmd)
        if client_api_pids is None or len(client_api_pids) <= 0:
            # Start local API services
            cur_dir = os.path.dirname(__file__)
            fedml_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(cur_dir)))
            python_program = get_python_program()
            self.local_api_process = ClientConstants.exec_console_with_script(
                "{} -m uvicorn {} --host 0.0.0.0 --port {} --reload --reload-delay 3 --reload-dir {} "
                "--log-level critical".format(
                    python_program, client_api_cmd,
                    ClientConstants.LOCAL_CLIENT_API_PORT, fedml_base_dir
                ),
                should_capture_stdout=False,
                should_capture_stderr=False
            )

    # Override
    def _process_connection_ready(self):
        MLOpsRuntimeLog.get_instance(self.args).init_logs(log_level=logging.INFO)

        if self.mqtt_inference_obj is None:
            self.mqtt_inference_obj = FedMLMqttInference(
                agent_config=self.agent_config, mqtt_mgr=self.communication_mgr)
        self.mqtt_inference_obj.setup_listener_for_endpoint_inference_request(self.edge_id)

    # Override
    def _process_connection_lost(self):
        try:
            if self.mqtt_inference_obj is not None:
                self.mqtt_inference_obj.remove_listener_for_endpoint_inference_request(self.edge_id)
        except Exception as e:
            pass

    # Override
    def print_connected_info(self):
        pass

    def callback_start_deployment(self, topic, payload):
        """
        topic: model_ops/model_device/start_deployment/model-agent-device-id
        payload: {"model_name": "image-model", "model_storage_url":"s3-url",
        "instance_scale_min":1, "instance_scale_max":3, "inference_engine":"onnx (or tensorrt)"}
        """
        # Parse deployment parameters
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
        scale_min = model_config.get("instance_scale_min", 0)
        scale_max = model_config.get("instance_scale_max", 0)
        inference_engine = model_config.get("inference_engine", 0)
        inference_end_point_id = run_id

        try:
            MLOpsConfigs.fetch_all_configs()
        except Exception as e:
            pass

        # Start log processor for current run
        run_id = inference_end_point_id
        self.args.run_id = run_id
        self.args.edge_id = self.edge_id
        MLOpsRuntimeLog(args=self.args).init_logs()
        MLOpsRuntimeLogDaemon.get_instance(self.args).set_log_source(
            ClientConstants.FEDML_LOG_SOURCE_TYPE_MODEL_END_POINT)
        MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(run_id, self.edge_id)

        # Start the job runner
        request_json["run_id"] = run_id
        run_id_str = str(run_id)
        self.request_json = request_json
        self.running_request_json[run_id_str] = request_json
        self._get_job_runner_manager().start_job_runner(
            run_id, request_json, args=self.args, edge_id=self.edge_id,
            sender_message_queue=self.message_center.get_sender_message_queue(),
            listener_message_queue=self.get_listener_message_queue(),
            status_center_queue=self.get_status_queue()
        )
        process = self._get_job_runner_manager().get_runner_process(run_id)
        if process is not None:
            ClientConstants.save_run_process(run_id, process.pid)

    def callback_delete_deployment(self, topic, payload):
        logging.info("[Worker] callback_delete_deployment")

        # Parse payload as the model message object.
        model_msg_object = FedMLModelMsgObject(topic, payload)

        # Delete all replicas on this device
        try:
            ClientConstants.remove_deployment(
                model_msg_object.end_point_name, model_msg_object.model_name, model_msg_object.model_version,
                model_msg_object.run_id, model_msg_object.model_id, edge_id=self.edge_id)
        except Exception as e:
            logging.info(f"Exception when removing deployment {traceback.format_exc()}")
            pass

        self._get_job_runner_manager().stop_job_runner(model_msg_object.run_id)

        logging.info(f"[endpoint/device][{model_msg_object.run_id}/{self.edge_id}] "
                     f"Release gpu resource when the worker deployment deleted.")
        JobRunnerUtils.get_instance().release_gpu_ids(model_msg_object.run_id, self.edge_id)

        if self.running_request_json.get(str(model_msg_object.run_id)) is not None:
            try:
                self.running_request_json.pop(str(model_msg_object.run_id))
            except Exception as e:
                logging.error(f"Error when removing running_request_json: {traceback.format_exc()}")
                pass

        FedMLClientDataInterface.get_instance().delete_job_from_db(model_msg_object.run_id)
        FedMLModelDatabase.get_instance().delete_deployment_result_with_device_id(
            model_msg_object.run_id, model_msg_object.end_point_name, model_msg_object.model_name,
            self.edge_id)

        # Delete FEDML_GLOBAL_ENDPOINT_RUN_ID_MAP_TAG-${run_id} both in redis and local db
        ComputeCacheManager.get_instance().gpu_cache.delete_endpoint_run_id_map(str(model_msg_object.run_id))

        # Delete FEDML_EDGE_ID_MODEL_DEVICE_ID_MAP_TAG-${run_id} both in redis and local db
        ComputeCacheManager.get_instance().gpu_cache.delete_edge_model_id_map(str(model_msg_object.run_id))

        # Delete FEDML_GLOBAL_DEVICE_RUN_GPU_IDS_TAG-${run_id}-${device_id} both in redis and local db
        ComputeCacheManager.get_instance().gpu_cache.delete_device_run_gpu_ids(str(self.edge_id),
                                                                               str(model_msg_object.run_id))

        # Delete FEDML_GLOBAL_DEVICE_RUN_NUM_GPUS_TAG-${run_id}-${device_id} both in redis and local db
        ComputeCacheManager.get_instance().gpu_cache.delete_device_run_num_gpus(str(self.edge_id),
                                                                                str(model_msg_object.run_id))

        # Delete FEDML_MODEL_REPLICA_GPU_IDS_TAG-${run_id}-${end_point_name}-${model_name}-${device_id}-*
        FedMLModelCache.get_instance().set_redis_params()
        FedMLModelCache.get_instance().delete_all_replica_gpu_ids(model_msg_object.run_id,
                                                                  model_msg_object.end_point_name,
                                                                  model_msg_object.model_name, self.edge_id)
