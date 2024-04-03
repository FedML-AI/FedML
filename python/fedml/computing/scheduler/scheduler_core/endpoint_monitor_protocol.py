import json

from .base_protocol import FedMLBaseProtocol
from ..model_scheduler.device_model_cache import FedMLModelCache
from ..model_scheduler.device_server_data_interface import FedMLServerDataInterface


class EndpointMsgConstants:
    SLAVE_MASTER_DEPLOYMENT_RESULT_MSG = "model_device/model_device/return_deployment_result"
    MASTER_MLOPS_DEPLOYMENT_RESULT_MSG = "model_ops/model_device/return_deployment_result"


class EndpointDeploymentResultModel:
    def __init__(self, payload):
        payload_json = json.loads(payload)
        self.endpoint_id = payload_json.get("end_point_id", None)
        self.endpoint_name = payload_json.get("end_point_name", None)
        self.model_name = payload_json.get("model_name", None)
        self.model_url = payload_json.get("model_url", None)
        self.version = payload_json.get("version", None)
        self.port = payload_json.get("port", None)
        self.inference_engine = payload_json.get("inference_engine", None)
        self.model_metadata = payload_json.get("model_metadata", None)
        self.model_config = payload_json.get("model_config", None)
        self.input_json = payload_json.get("input_json", None)
        self.output_json = payload_json.get("output_json", None)
        self.timestamp = payload_json.get("timestamp", None)


class EndpointDeviceDeploymentResultModel:
    def __init__(self, payload):
        payload_json = json.loads(payload)
        self.endpoint_id = payload_json.get("end_point_id", None)
        self.endpoint_name = payload_json.get("end_point_name", None)
        self.model_id = payload_json.get("model_id", None)
        self.model_name = payload_json.get("model_name", None)
        self.model_url = payload_json.get("model_url", None)
        self.model_version = payload_json.get("model_version", None)
        self.port = payload_json.get("port", None)
        self.inference_engine = payload_json.get("inference_engine", None)
        self.model_metadata = payload_json.get("model_metadata", None)
        self.model_config = payload_json.get("model_config", None)
        self.model_status = payload_json.get("model_status", None)
        self.inference_port = payload_json.get("inference_port", None)


class EndpointDeviceDeploymentStatusModel:
    def __init__(self, payload):
        payload_json = json.loads(payload)
        self.endpoint_id = payload_json.get("end_point_id", None)
        self.endpoint_name = payload_json.get("end_point_name", None)
        self.model_id = payload_json.get("model_id", None)
        self.model_name = payload_json.get("model_name", None)
        self.model_url = payload_json.get("model_url", None)
        self.model_version = payload_json.get("model_version", None)
        self.model_status = payload_json.get("model_status", None)
        self.inference_port = payload_json.get("inference_port", None)
        self.device_id = payload_json.get("device_id", None)


class EndpointDeviceDeploymentInfoModel:
    def __init__(self, payload):
        payload_json = json.loads(payload)
        self.endpoint_id = payload_json.get("end_point_id", None)
        self.endpoint_name = payload_json.get("end_point_name", None)
        self.model_id = payload_json.get("model_id", None)
        self.model_name = payload_json.get("model_name", None)
        self.model_version = payload_json.get("model_version", None)
        self.inference_port = payload_json.get("inference_port", None)
        self.device_id = payload_json.get("device_id", None)
        self.disable = payload_json.get("disable", None)
        self.replica_no = payload_json.get("replica_no", None)


class FedMLEndpointMonitorProtocol(FedMLBaseProtocol):

    def __init__(self, agent_config=None, mqtt_mgr=None):
        super().__init__(agent_config=agent_config, mqtt_mgr=mqtt_mgr)
        self.master_id = None

    def setup_listener_for_endpoint_result(self, endpoint_id):
        deployment_results_topic = f"{EndpointMsgConstants.MASTER_MLOPS_DEPLOYMENT_RESULT_MSG}/{endpoint_id}"
        self.client_mqtt_mgr.add_message_listener(deployment_results_topic, self.callback_deployment_result)
        self.client_mqtt_mgr.subscribe_msg(deployment_results_topic)

    def remove_listener_for_endpoint_result(self, endpoint_id):
        deployment_results_topic = f"{EndpointMsgConstants.MASTER_MLOPS_DEPLOYMENT_RESULT_MSG}/{endpoint_id}"
        self.client_mqtt_mgr.remove_message_listener(deployment_results_topic)
        self.client_mqtt_mgr.unsubscribe_msg(deployment_results_topic)

    def setup_listener_for_device_result(self, device_id):
        deployment_results_topic = f"{EndpointMsgConstants.SLAVE_MASTER_DEPLOYMENT_RESULT_MSG}/{device_id}"
        self.client_mqtt_mgr.add_message_listener(
            EndpointMsgConstants.SLAVE_MASTER_DEPLOYMENT_RESULT_MSG, self.callback_device_result)
        self.client_mqtt_mgr.subscribe_msg(deployment_results_topic)

    def remove_listener_for_device_result(self, device_id):
        deployment_results_topic = f"{EndpointMsgConstants.SLAVE_MASTER_DEPLOYMENT_RESULT_MSG}/{device_id}"
        self.client_mqtt_mgr.remove_message_listener(deployment_results_topic)
        self.client_mqtt_mgr.unsubscribe_msg(deployment_results_topic)

    def callback_deployment_result(self, topic, payload):
        endpoint_result_model = EndpointDeploymentResultModel(payload)

    def callback_device_result(self, topic, payload):
        # Save deployment result to local cache
        topic_splits = str(topic).split('/')
        device_id = topic_splits[-1]
        deployment_result = EndpointDeviceDeploymentResultModel(payload)
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port).set_deployment_result(
            deployment_result.endpoint_id, deployment_result.endpoint_name, deployment_result.model_name,
            deployment_result.model_version, device_id, payload)

        payload_json_saved = json.loads(payload)
        payload_json_saved["model_slave_url"] = payload_json_saved["model_url"]
        FedMLServerDataInterface.get_instance().save_job_result(
            deployment_result.endpoint_id, self.master_id, json.dumps(payload_json_saved))
