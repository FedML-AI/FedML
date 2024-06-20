
import json
import logging
from fedml.core.mlops import MLOpsConfigs, MLOpsRuntimeLog, MLOpsRuntimeLogDaemon
from .device_model_cache import FedMLModelCache
from .device_model_db import FedMLModelDatabase
from .device_model_msg_object import FedMLModelMsgObject
from .device_server_constants import ServerConstants
from .device_server_data_interface import FedMLServerDataInterface
from ..master.base_master_protocol_manager import FedMLBaseMasterProtocolManager
from .master_job_runner_manager import FedMLDeployJobRunnerManager
from ..scheduler_core.general_constants import GeneralConstants
from ..scheduler_core.endpoint_sync_protocol import FedMLEndpointSyncProtocol
from ..scheduler_core.compute_cache_manager import ComputeCacheManager


class FedMLDeployMasterProtocolManager(FedMLBaseMasterProtocolManager):
    def __init__(self, args, agent_config=None):
        FedMLBaseMasterProtocolManager.__init__(self, args, agent_config=agent_config)

        self.message_center_name = "deploy_master_agent"
        self.is_deployment_status_center = True

        self.topic_start_deployment = None
        self.topic_activate_endpoint = None
        self.topic_deactivate_deployment = None
        self.topic_delete_deployment = None

        self.infer_host = "127.0.0.1"
        self.redis_addr = "local"
        self.redis_port = "6379"
        self.redis_password = "fedml_default"
        self.endpoint_sync_protocol = None

    # Override
    def _generate_protocol_manager_instance(self, args, agent_config=None):
        return FedMLDeployMasterProtocolManager(args, agent_config=agent_config)

    # Override
    def generate_topics(self):
        super().generate_topics()

        # The topic for start deployment
        self.topic_start_deployment = "model_ops/model_device/start_deployment/{}".format(str(self.edge_id))

        # The topic for activating endpoint
        self.topic_activate_endpoint = "model_ops/model_device/activate_deployment/{}".format(str(self.edge_id))

        # The topic for activating endpoint
        self.topic_deactivate_deployment = "model_ops/model_device/deactivate_deployment/{}".format(str(self.edge_id))

        # The topic for deleting endpoint
        self.topic_delete_deployment = "model_ops/model_device/delete_deployment/{}".format(str(self.edge_id))

        # Subscribe topics for endpoints
        self.add_subscribe_topic(self.topic_start_deployment)
        self.add_subscribe_topic(self.topic_activate_endpoint)
        self.add_subscribe_topic(self.topic_deactivate_deployment)
        self.add_subscribe_topic(self.topic_delete_deployment)

    # Override
    def add_protocol_handler(self):
        super().add_protocol_handler()

        # Add the message listeners for endpoint related topics
        self.add_message_listener(self.topic_start_deployment, self.callback_start_deployment)
        self.add_message_listener(self.topic_activate_endpoint, self.callback_activate_deployment)
        self.add_message_listener(self.topic_deactivate_deployment, self.callback_deactivate_deployment)
        self.add_message_listener(self.topic_delete_deployment, self.callback_delete_deployment)

    # Override
    def _get_job_runner_manager(self):
        return FedMLDeployJobRunnerManager.get_instance()

    # Override
    def _init_extra_items(self):
        # Init local database
        FedMLServerDataInterface.get_instance().create_job_table()
        try:
            FedMLModelDatabase.get_instance().set_database_base_dir(ServerConstants.get_database_dir())
            FedMLModelDatabase.get_instance().create_table()
        except Exception as e:
            pass

        FedMLDeployJobRunnerManager.recover_inference_and_monitor()

    # Override
    def _process_connection_ready(self):
        self.endpoint_sync_protocol = FedMLEndpointSyncProtocol(
            agent_config=self.agent_config, mqtt_mgr=self.message_center)
        self.endpoint_sync_protocol.setup_listener_for_sync_device_info(self.edge_id)

        MLOpsRuntimeLog.get_instance(self.args).init_logs(log_level=logging.INFO)

    # Override
    def _process_connection_lost(self):
        pass

    # Override
    def print_connected_info(self):
        pass

    def callback_deployment_result_message(self, topic=None, payload=None):
        logging.info(f"Received deployment result")
        FedMLDeployJobRunnerManager.get_instance().save_deployment_result(topic, payload)

    def callback_delete_deployment(self, topic, payload):
        logging.info("[Master] callback_delete_deployment")
        # Parse payload as the model message object.
        model_msg_object = FedMLModelMsgObject(topic, payload)

        # Get the launch job id
        ComputeCacheManager.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        launch_job_id = ComputeCacheManager.get_instance().get_gpu_cache().get_endpoint_run_id_map(model_msg_object.run_id)

        # Delete SQLite records
        FedMLServerDataInterface.get_instance().delete_job_from_db(model_msg_object.run_id)
        FedMLModelDatabase.get_instance().delete_deployment_result(
            model_msg_object.run_id, model_msg_object.end_point_name, model_msg_object.model_name,
            model_version=model_msg_object.model_version)
        FedMLModelDatabase.get_instance().delete_deployment_run_info(
            end_point_id=model_msg_object.inference_end_point_id)

        # Delete Redis Records
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            set_end_point_activation(model_msg_object.inference_end_point_id,
                                     model_msg_object.end_point_name, False)
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            delete_end_point(model_msg_object.inference_end_point_id, model_msg_object.end_point_name,
                             model_msg_object.model_name, model_msg_object.model_version)

        # Send delete deployment request to the edge devices
        FedMLDeployJobRunnerManager.get_instance().send_deployment_delete_request_to_edges(
            model_msg_object.run_id, payload, model_msg_object, message_center=self.message_center)

        # Stop processes on master
        FedMLDeployJobRunnerManager.get_instance().stop_job_runner(model_msg_object.run_id)
        FedMLDeployJobRunnerManager.get_instance().stop_device_inference_monitor(
            model_msg_object.run_id, model_msg_object.end_point_name, model_msg_object.model_id,
            model_msg_object.model_name, model_msg_object.model_version)

        # Report the launch job status with killed status.
        if launch_job_id is not None:
            self.generate_status_report(model_msg_object.run_id, self.edge_id, server_agent_id=self.edge_id).\
                report_server_id_status(launch_job_id, GeneralConstants.MSG_MLOPS_SERVER_STATUS_KILLED,
                                        server_id=self.edge_id, server_agent_id=self.edge_id)

    def callback_start_deployment(self, topic, payload):
        # noinspection PyBroadException
        try:
            MLOpsConfigs.fetch_all_configs()
        except Exception as e:
            pass

        # Get deployment params
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
        model_version = model_config["model_version"]
        model_id = model_config["model_id"]
        model_storage_url = model_config["model_storage_url"]
        scale_min = model_config.get("instance_scale_min", 0)
        scale_max = model_config.get("instance_scale_max", 0)
        inference_engine = model_config.get("inference_engine", 0)
        enable_auto_scaling = request_json.get("enable_auto_scaling", False)
        desired_replica_num = request_json.get("desired_replica_num", 1)

        target_queries_per_replica = request_json.get("target_queries_per_replica", 10)
        aggregation_window_size_seconds = request_json.get("aggregation_window_size_seconds", 60)
        scale_down_delay_seconds = request_json.get("scale_down_delay_seconds", 120)

        model_config_parameters = request_json.get("parameters", {})
        timeout_s = model_config_parameters.get("request_timeout_sec", 30)

        inference_end_point_id = run_id

        logging.info("[Master] received start deployment request for end point {}.".format(run_id))

        # Set redis config
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)

        # Query if the endpoint exists
        endpoint_device_info = FedMLModelCache.get_instance(self.redis_addr, self.redis_port).get_end_point_device_info(
            request_json["end_point_id"])
        request_json["is_fresh_endpoint"] = True if endpoint_device_info is None else False

        # Save the user setting (about replica number) of this run to Redis, if existed, update it
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port).set_user_setting_replica_num(
            end_point_id=run_id, end_point_name=end_point_name, model_name=model_name, model_version=model_version,
            replica_num=desired_replica_num, enable_auto_scaling=enable_auto_scaling,
            scale_min=scale_min, scale_max=scale_max, state="DEPLOYING",
            aggregation_window_size_seconds=aggregation_window_size_seconds,
            target_queries_per_replica=target_queries_per_replica,
            scale_down_delay_seconds=int(scale_down_delay_seconds),
            timeout_s=timeout_s
        )

        # Start log processor for current run
        self.args.run_id = run_id
        self.args.edge_id = self.edge_id
        MLOpsRuntimeLog(args=self.args).init_logs()
        MLOpsRuntimeLogDaemon.get_instance(self.args).set_log_source(
            ServerConstants.FEDML_LOG_SOURCE_TYPE_MODEL_END_POINT)
        MLOpsRuntimeLogDaemon.get_instance(self.args).start_log_processor(run_id, self.edge_id)

        # Add additional parameters to the request_json
        run_id = inference_end_point_id
        self.args.run_id = run_id
        self.run_id = run_id
        request_json["run_id"] = run_id
        self.request_json = request_json
        run_id_str = str(run_id)
        self.running_request_json[run_id_str] = request_json
        self.request_json["master_node_ip"] = GeneralConstants.get_ip_address(request_json)

        # Set the target status of the devices to redis
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            set_end_point_device_info(request_json["end_point_id"], end_point_name, json.dumps(device_objs))

        # Setup Token
        usr_indicated_token = self.get_usr_indicated_token(request_json)
        if usr_indicated_token != "":
            logging.info(f"Change Token from{token} to {usr_indicated_token}")
            token = usr_indicated_token
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            set_end_point_token(run_id, end_point_name, model_name, token)

        self.subscribe_deployment_messages_from_slave_devices(request_json)

        ServerConstants.save_runner_infos(self.args.device_id + "." + self.args.os_name, self.edge_id, run_id=run_id)

        # Num diff
        request_json = FedMLDeployJobRunnerManager.generate_request_json_with_replica_num_diff(
            run_id, self.edge_id, request_json
        )

        # Listen to extra worker topics, especially when worker's replica remove to zero,
        # In this case, currently Java will NOT send those worker ids to the master, but still need to listen to it.
        if "replica_num_diff" in request_json and len(request_json["replica_num_diff"]) > 0:
            for device_id in request_json["replica_num_diff"].keys():
                # {"op": "remove", "curr_num": 1, "target_num": 0}
                if request_json["replica_num_diff"][device_id]["op"] == "remove" and \
                        request_json["replica_num_diff"][device_id]["target_num"] == 0:
                    self.subscribe_spec_device_message(run_id, device_id)

        # Version diff
        request_json = FedMLDeployJobRunnerManager.generate_request_json_with_replica_version_diff(
            run_id, self.edge_id, request_json
        )
        self.running_request_json[run_id_str] = request_json

        # Start the job runner to deploy models
        self._get_job_runner_manager().start_job_runner(
            run_id, request_json, args=self.args, edge_id=self.edge_id,
            sender_message_queue=self.message_center.get_sender_message_queue(),
            listener_message_queue=self.get_listener_message_queue(),
            status_center_queue=self.get_status_queue(),
            process_name=GeneralConstants.get_deploy_master_job_process_name(run_id, self.edge_id)
        )
        process = self._get_job_runner_manager().get_runner_process(run_id)
        if process is not None:
            ServerConstants.save_run_process(run_id, process.pid)

        # Report stage to mlops: MODEL_DEPLOYMENT_STAGE1 = "Received"
        FedMLDeployJobRunnerManager.get_instance().send_deployment_stages(
            run_id, model_name, model_id, "", ServerConstants.MODEL_DEPLOYMENT_STAGE1["index"],
            ServerConstants.MODEL_DEPLOYMENT_STAGE1["text"], "Received request for endpoint {}".format(run_id),
            message_center=self.message_center)

        # Report stage to mlops: MODEL_DEPLOYMENT_STAGE2 = "Initializing"
        FedMLDeployJobRunnerManager.get_instance().send_deployment_stages(
            run_id, model_name, model_id, "", ServerConstants.MODEL_DEPLOYMENT_STAGE2["index"],
            ServerConstants.MODEL_DEPLOYMENT_STAGE2["text"], ServerConstants.MODEL_DEPLOYMENT_STAGE2["text"],
            message_center=self.message_center)

        # Send stage: MODEL_DEPLOYMENT_STAGE3 = "StartRunner"
        FedMLDeployJobRunnerManager.get_instance().send_deployment_stages(
            run_id, model_name, model_id, "", ServerConstants.MODEL_DEPLOYMENT_STAGE3["index"],
            ServerConstants.MODEL_DEPLOYMENT_STAGE3["text"], ServerConstants.MODEL_DEPLOYMENT_STAGE3["text"],
            message_center=self.message_center)

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

    @staticmethod
    def get_usr_indicated_token(request_json) -> str:
        usr_indicated_token = ""
        if "parameters" in request_json and "authentication_token" in request_json["parameters"]:
            usr_indicated_token = request_json["parameters"]["authentication_token"]
        return usr_indicated_token

    def init_device_update_map(self):
        # [Deprecated] Use the replica controller to manage the device update
        pass

    def subscribe_deployment_messages_from_slave_devices(self, request_json):
        if request_json is None:
            return
        run_id = request_json["run_id"]
        edge_id_list = request_json["device_ids"]
        logging.info("Edge ids: " + str(edge_id_list))
        for edge_id in edge_id_list:
            if str(edge_id) == str(self.edge_id):
                continue
            # subscribe deployment result message for each model device
            deployment_results_topic = "model_device/model_device/return_deployment_result/{}/{}".format(
                run_id, edge_id)
            self.add_message_listener(deployment_results_topic, self.callback_deployment_result_message)
            self.subscribe_msg(deployment_results_topic)

            logging.info("subscribe device messages {}".format(deployment_results_topic))

        self.setup_listeners_for_edge_status(run_id, edge_id_list, self.edge_id)

    def subscribe_spec_device_message(self, run_id, device_id):
        if device_id == self.edge_id:
            return

        # subscribe deployment result message for each model device
        deployment_results_topic = "model_device/model_device/return_deployment_result/{}/{}".format(
            run_id, device_id)

        self.add_message_listener(deployment_results_topic, self.callback_deployment_result_message)
        self.subscribe_msg(deployment_results_topic)
