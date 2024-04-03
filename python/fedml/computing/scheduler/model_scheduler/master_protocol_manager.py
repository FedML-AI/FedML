
import json
import logging
import os
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


class FedMLDeployMasterProtocolManager(FedMLBaseMasterProtocolManager):
    def __init__(self, args, agent_config=None):
        FedMLBaseMasterProtocolManager.__init__(self, args, agent_config=agent_config)

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
        logging.info(f"Received deployment result: {self}")
        FedMLDeployJobRunnerManager.get_instance().save_deployment_result(topic, payload)

    def callback_delete_deployment(self, topic, payload):
        # Parse payload as the model message object.
        logging.info("[Master] callback_delete_deployment")
        model_msg_object = FedMLModelMsgObject(topic, payload)

        # Set end point as deactivated status
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            set_end_point_activation(model_msg_object.inference_end_point_id,
                                     model_msg_object.end_point_name, False)
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            delete_end_point(model_msg_object.inference_end_point_id, model_msg_object.end_point_name,
                             model_msg_object.model_name, model_msg_object.model_version)

        FedMLDeployJobRunnerManager.get_instance().send_deployment_delete_request_to_edges(
            model_msg_object.inference_end_point_id, payload, model_msg_object)

        FedMLDeployJobRunnerManager.get_instance().stop_job_runner(model_msg_object.run_id)

        FedMLDeployJobRunnerManager.get_instance().stop_device_inference_monitor(
            model_msg_object.run_id, model_msg_object.end_point_name, model_msg_object.model_id,
            model_msg_object.model_name, model_msg_object.model_version)

        FedMLServerDataInterface.get_instance().delete_job_from_db(model_msg_object.run_id)
        FedMLModelDatabase.get_instance().delete_deployment_result(
            model_msg_object.run_id, model_msg_object.end_point_name, model_msg_object.model_name,
            model_version=model_msg_object.model_version)
        FedMLModelDatabase.get_instance().delete_deployment_run_info(
            end_point_id=model_msg_object.inference_end_point_id)

    def callback_start_deployment(self, topic, payload):
        # noinspection PyBroadException
        try:
            MLOpsConfigs.fetch_all_configs()
        except Exception as e:
            pass

        # Parse the deployment parameters
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

        # Generate the deployment new parameters
        logging.info("callback_start_deployment {}".format(payload))
        run_id = inference_end_point_id
        run_id_str = str(run_id)
        request_json["run_id"] = run_id
        self.request_json = request_json
        self.running_request_json[run_id_str] = request_json
        diff_devices, diff_version = self.get_diff_devices(run_id)
        self.request_json["diff_devices"] = diff_devices
        self.request_json["diff_version"] = diff_version
        self.request_json["master_node_ip"] = GeneralConstants.get_ip_address(self.request_json)

        # Save the endpoint device info
        self.init_device_update_map()
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            set_end_point_device_info(request_json["end_point_id"], end_point_name, json.dumps(device_objs))

        # Save the endpoint token
        usr_indicated_token = FedMLDeployMasterProtocolManager.get_usr_indicated_token(request_json)
        if usr_indicated_token != "":
            logging.info(f"Change Token from{token} to {usr_indicated_token}")
            token = usr_indicated_token
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            set_end_point_token(run_id, end_point_name, model_name, token)

        # Subscribe deployment result messages from slave devices
        self.subscribe_deployment_messages_from_slave_devices(request_json)

        # Send stage: MODEL_DEPLOYMENT_STAGE1 = "Received"
        FedMLDeployJobRunnerManager.get_instance().send_deployment_stages(
            self.run_id, model_name, model_id, "", ServerConstants.MODEL_DEPLOYMENT_STAGE1["index"],
            ServerConstants.MODEL_DEPLOYMENT_STAGE1["text"], "Received request for end point {}".format(run_id),
            message_center=self.message_center)

        # Send stage: MODEL_DEPLOYMENT_STAGE2 = "Initializing"
        FedMLDeployJobRunnerManager.get_instance().send_deployment_stages(
            self.run_id, model_name, model_id, "", ServerConstants.MODEL_DEPLOYMENT_STAGE2["index"],
            ServerConstants.MODEL_DEPLOYMENT_STAGE2["text"], ServerConstants.MODEL_DEPLOYMENT_STAGE2["text"],
            message_center=self.message_center)

        # Save the runner info
        ServerConstants.save_runner_infos(self.args.device_id + "." + self.args.os_name, self.edge_id, run_id=run_id)

        # Start the job runner to deploy models
        self.running_request_json[run_id_str] = FedMLDeployJobRunnerManager.generate_request_json_with_replica_diff(
            run_id, self.edge_id, request_json
        )
        self._get_job_runner_manager().start_job_runner(
            run_id, request_json, args=self.args, edge_id=self.edge_id,
            sender_message_queue=self.message_center.get_sender_message_queue(),
            listener_message_queue=self.get_listener_message_queue(),
            status_center_queue=self.get_status_queue()
        )
        process = self._get_job_runner_manager().get_runner_process(run_id)
        if process is not None:
            ServerConstants.save_run_process(run_id, process.pid)

        # Send stage: MODEL_DEPLOYMENT_STAGE3 = "StartRunner"
        FedMLDeployJobRunnerManager.get_instance().send_deployment_stages(
            self.run_id, model_name, model_id, "", ServerConstants.MODEL_DEPLOYMENT_STAGE3["index"],
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

    def get_diff_devices(self, run_id) -> (dict, dict):
        """
        {device_id(int): "op: add" | "op: delete" | "op: replace"}
        "op: add" -> need to add
        "op: delete" -> need to delete device
        "op: replace" -> need to restart the container of the device on same port with new (same) model pkg

        {device_id(int): "old_version"}
        """
        try:
            logging.info(f"Get diff devices for run {run_id}")
            request_json = self.running_request_json.get(str(run_id))

            diff_devices = {}
            diff_version = {}
            FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
            device_objs = FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
                get_end_point_device_info(run_id)
            if device_objs is None:
                for new_device_id in request_json["device_ids"]:
                    diff_devices[new_device_id] = ServerConstants.DEVICE_DIFF_ADD_OPERATION
            else:
                device_objs_dict = json.loads(device_objs)
                device_ids_frm_db = [d["id"] for d in device_objs_dict]

                for exist_device_id in device_ids_frm_db:
                    if exist_device_id not in request_json["device_ids"]:
                        diff_devices[exist_device_id] = ServerConstants.DEVICE_DIFF_DELETE_OPERATION

                for new_device_id in request_json["device_ids"]:
                    if new_device_id not in device_ids_frm_db:
                        diff_devices[new_device_id] = ServerConstants.DEVICE_DIFF_ADD_OPERATION
                    else:
                        if new_device_id == self.edge_id:
                            continue

                        old_version = self.should_update_device(request_json, new_device_id)
                        if old_version:
                            diff_devices[new_device_id] = ServerConstants.DEVICE_DIFF_REPLACE_OPERATION
                            diff_version[new_device_id] = old_version
                        else:
                            pass
            logging.info(f"Diff devices: {diff_devices}")
        except Exception as e:
            error_log_path = f"~/.fedml/fedml-model-server/fedml/logs/{run_id}_error.txt"
            if not os.path.exists(os.path.dirname(os.path.expanduser(error_log_path))):
                os.makedirs(os.path.dirname(os.path.expanduser(error_log_path)))
            with open(os.path.expanduser(error_log_path), "w") as f:
                f.write(str(e))
            raise e
        return diff_devices, diff_version

    def should_update_device(self, payload, new_device_id):
        """
        Query the device info in local redis, if the device info is different from the payload,
        return the old model version
        """
        device_result_list = FedMLModelCache.get_instance(self.redis_addr, self.redis_port). \
            get_deployment_result_list(self.request_json["end_point_id"],
                                       self.request_json["end_point_name"],
                                       self.request_json["model_config"]["model_name"])

        for device_result in device_result_list:
            if device_result is None:
                continue
            device_result_dict = json.loads(device_result)

            if int(device_result_dict["cache_device_id"]) == new_device_id:
                result_body = json.loads(device_result_dict["result"])
                if result_body["model_version"] != payload["model_config"]["model_version"]:
                    return result_body["model_version"]
                else:
                    return None
        return None

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
            deployment_results_topic = "model_device/model_device/return_deployment_result/{}".format(edge_id)
            self.add_message_listener(deployment_results_topic, self.callback_deployment_result_message)
            self.subscribe_msg(deployment_results_topic)

            logging.info("subscribe device messages {}".format(deployment_results_topic))
