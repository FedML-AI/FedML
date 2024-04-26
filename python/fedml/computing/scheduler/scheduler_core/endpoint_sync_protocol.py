import json

from .base_protocol import FedMLBaseProtocol
from ..model_scheduler.device_model_cache import FedMLModelCache
from ..model_scheduler.device_model_db import FedMLModelDatabase
from ..model_scheduler.device_server_data_interface import FedMLServerDataInterface
from .endpoint_monitor_protocol import EndpointDeviceDeploymentResultModel, EndpointDeviceDeploymentStatusModel, EndpointDeviceDeploymentInfoModel
from ..model_scheduler.device_server_constants import ServerConstants
from urllib.parse import urlparse
import logging


class EndpointSyncMsgConstants:
    SLAVE_MASTER_DEPLOYMENT_RESULT_MSG = "model_device/model_device/sync_deployment_result"
    SLAVE_MASTER_DEPLOYMENT_STATUS_MSG = "model_device/model_device/sync_deployment_status"
    SLAVE_MASTER_DEPLOYMENT_INFO_MSG = "model_device/model_device/sync_deployment_info"


class FedMLEndpointSyncProtocol(FedMLBaseProtocol):

    def __init__(self, agent_config=None, mqtt_mgr=None):
        super().__init__(agent_config=agent_config, mqtt_mgr=mqtt_mgr)
        self.master_id = None

    def setup_listener_for_sync_device_result(self, device_id):
        deployment_results_topic = f"{EndpointSyncMsgConstants.SLAVE_MASTER_DEPLOYMENT_RESULT_MSG}/{device_id}"
        self.client_mqtt_mgr.add_message_listener(deployment_results_topic, self.callback_sync_device_result)
        self.client_mqtt_mgr.subscribe_msg(deployment_results_topic)

    def remove_listener_for_sync_device_result(self, device_id):
        deployment_results_topic = f"{EndpointSyncMsgConstants.SLAVE_MASTER_DEPLOYMENT_RESULT_MSG}/{device_id}"
        self.client_mqtt_mgr.remove_message_listener(deployment_results_topic)
        self.client_mqtt_mgr.unsubscribe_msg(deployment_results_topic)

    def setup_listener_for_sync_device_status(self, device_id):
        deployment_status_topic = f"{EndpointSyncMsgConstants.SLAVE_MASTER_DEPLOYMENT_STATUS_MSG}/{device_id}"
        self.client_mqtt_mgr.add_message_listener(deployment_status_topic, self.callback_sync_device_status)
        self.client_mqtt_mgr.subscribe_msg(deployment_status_topic)

    def remove_listener_for_sync_device_status(self, device_id):
        deployment_status_topic = f"{EndpointSyncMsgConstants.SLAVE_MASTER_DEPLOYMENT_STATUS_MSG}/{device_id}"
        self.client_mqtt_mgr.remove_message_listener(deployment_status_topic)
        self.client_mqtt_mgr.unsubscribe_msg(deployment_status_topic)

    def setup_listener_for_sync_device_info(self, device_id):
        deployment_info_topic = f"{EndpointSyncMsgConstants.SLAVE_MASTER_DEPLOYMENT_INFO_MSG}/{device_id}"
        self.client_mqtt_mgr.add_message_listener(deployment_info_topic, self.callback_sync_device_info)
        self.client_mqtt_mgr.subscribe_msg(deployment_info_topic)

    def remove_listener_for_sync_device_info(self, device_id):
        deployment_info_topic = f"{EndpointSyncMsgConstants.SLAVE_MASTER_DEPLOYMENT_INFO_MSG}/{device_id}"
        self.client_mqtt_mgr.remove_message_listener(deployment_info_topic)
        self.client_mqtt_mgr.unsubscribe_msg(deployment_info_topic)

    def send_sync_deployment_results(self, master_id, result_payload):
        deployment_results_topic = "model_device/model_device/sync_deployment_result/{}".format(master_id)
        logging.info("send_sync_deployment_results: topic {}, payload {}.".format(
            deployment_results_topic, result_payload))
        self.client_mqtt_mgr.send_message_json(deployment_results_topic, json.dumps(result_payload))
        return result_payload

    def send_sync_deployment_status(self, master_id, status_payload):
        deployment_status_topic = "model_device/model_device/sync_deployment_status/{}".format(master_id)
        logging.info("send_sync_deployment_status: topic {}, payload {}.".format(
            deployment_status_topic, status_payload))
        self.client_mqtt_mgr.send_message_json(deployment_status_topic, json.dumps(status_payload))

    def send_sync_inference_info(
            self, master_id, worker_device_id, end_point_id, end_point_name, model_name,
            model_id, model_version, inference_port, replica_no, disable=False):
        deployment_info_topic = f"{EndpointSyncMsgConstants.SLAVE_MASTER_DEPLOYMENT_INFO_MSG}/{master_id}"
        deployment_info_payload = {
            "device_id": worker_device_id, "end_point_id": end_point_id, "end_point_name": end_point_name,
            "model_id": model_id, "model_name": model_name, "model_version": model_version,
            "inference_port": inference_port, "disable": disable, "replica_no": replica_no}
        self.client_mqtt_mgr.send_message_json(deployment_info_topic, json.dumps(deployment_info_payload))

    def callback_sync_device_result(self, topic, payload):
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

    def callback_sync_device_status(self, topic, payload):
        # Save deployment status to local cache
        topic_splits = str(topic).split('/')
        device_id = topic_splits[-1]
        deployment_status = EndpointDeviceDeploymentStatusModel(payload)
        FedMLModelCache.get_instance().set_redis_params(self.redis_addr, self.redis_port, self.redis_password)
        FedMLModelCache.get_instance(self.redis_addr, self.redis_port).set_deployment_status(
            deployment_status.endpoint_id, deployment_status.endpoint_name, deployment_status.model_name,
            deployment_status.model_version, device_id, deployment_status.model_status)

        payload_json_saved = json.loads(payload)
        payload_json_saved["model_slave_url"] = payload_json_saved["model_url"]
        FedMLServerDataInterface.get_instance().save_job_status(
            deployment_status.endpoint_id, self.master_id, deployment_status.model_status,
            deployment_status.model_status
        )

    def callback_sync_device_info(self, topic, payload):
        # Save deployment info to local cache
        topic_splits = str(topic).split('/')
        master_device_id = topic_splits[-1]
        deployment_info = EndpointDeviceDeploymentInfoModel(payload)

        # Status [Deprecated]

        # Find the existed result in the local db
        result_item_found = None
        result_payload_found = None
        FedMLModelCache.get_instance().set_redis_params()
        result_list = FedMLModelCache.get_instance().get_deployment_result_list(
            deployment_info.endpoint_id, deployment_info.endpoint_name, deployment_info.model_name)
        for result_item in result_list:
            cache_device_id, replica_no, result_payload = (
                FedMLModelCache.get_instance().get_result_item_info(result_item))
            if str(cache_device_id) == str(deployment_info.device_id) and replica_no == deployment_info.replica_no:
                result_item_found = result_item
                result_payload_found = result_payload
                break

        # Update the replica element
        if result_item_found is not None:
            # print(f"result_item_found {result_item_found}, result_payload_found {result_payload_found}")

            # Delete the replica info from the local redis
            # TODO: Delete also in the sqlite
            FedMLModelCache.get_instance().delete_deployment_result(
                result_item_found, deployment_info.endpoint_id, deployment_info.endpoint_name,
                deployment_info.model_name)
            
            if deployment_info.disable:
                # This replica is down
                result_payload_found["model_status"] = ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_UPDATING
            else:
                result_payload_found["model_status"] = ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED
            
            model_url_parsed = urlparse(result_payload_found.get("model_url", ""))
            result_payload_found["model_url"] = (f"http://{model_url_parsed.hostname}:{deployment_info.inference_port}"
                                                 f"{model_url_parsed.path}")
            result_payload_found["inference_port"] = deployment_info.inference_port
            FedMLModelCache.get_instance().set_deployment_result(
                deployment_info.endpoint_id, deployment_info.endpoint_name, deployment_info.model_name,
                deployment_info.model_version, deployment_info.device_id, json.dumps(result_payload_found),
                deployment_info.replica_no)

    def set_local_deployment_status_result(
            self, endpoint_id, endpoint_name, model_name, model_version, device_id,
            inference_port, status_payload, result_payload, replica_no):
        """
        The result and status are saved in the local sqlite table.
        They both belong to the table deployment_result_info;
        deployment_result column is used to save the result;
        deployment_status column is used to save the status.
        """
        # status_payload is deprecated
        # result_payload
        if result_payload is not None:
            model_url_parsed = urlparse(result_payload.get("model_url", ""))
            result_payload["model_url"] = f"http://{model_url_parsed.hostname}:{inference_port}{model_url_parsed.path}"
            result_payload["inference_port"] = inference_port
            FedMLModelDatabase.get_instance().set_deployment_result(
                endpoint_id, endpoint_name, model_name, model_version=None, device_id=device_id,
                deployment_result=json.dumps(result_payload), replica_no=replica_no)
