
import json
import logging
import os
import time
from .device_model_cache import FedMLModelCache
from .device_server_constants import ServerConstants
from ..scheduler_core.general_constants import GeneralConstants


class FedMLDeployJobRunnerMsgSender(object):
    def __init__(self):
        self.infer_host = "127.0.0.1"
        self.redis_addr = "local"
        self.redis_port = "6379"
        self.redis_password = "fedml_default"
        self.message_center = None
        self.request_json = None
        self.edge_id = None

    def send_deployment_results_with_payload(self, end_point_id, end_point_name, payload, replica_id_list=None):
        self.send_deployment_results(end_point_id, end_point_name,
                                     payload["model_name"], payload["model_url"],
                                     payload["model_version"], payload["port"],
                                     payload["inference_engine"],
                                     payload["model_metadata"],
                                     payload["model_config"],
                                     payload["input_json"],
                                     payload["output_json"],
                                     replica_id_list=replica_id_list)

    def send_deployment_results(self, end_point_id, end_point_name,
                                model_name, model_inference_url,
                                model_version, inference_port, inference_engine,
                                model_metadata, model_config, input_json, output_json, replica_id_list=None):
        deployment_results_topic = "model_ops/model_device/return_deployment_result"
        deployment_results_payload = {"end_point_id": end_point_id, "end_point_name": end_point_name,
                                      "model_name": model_name, "model_url": model_inference_url,
                                      "version": model_version, "port": inference_port,
                                      "inference_engine": inference_engine,
                                      "model_metadata": model_metadata,
                                      "model_config": model_config,
                                      "input_json": input_json,
                                      "output_json": output_json,
                                      "timestamp": int(format(time.time_ns() / 1000.0, '.0f')),
                                      "replica_ids": replica_id_list}
        logging.info(f"[Master] deployment_results_payload is sent to mlops: {deployment_results_payload}")

        self.message_center.send_message_json(deployment_results_topic, json.dumps(deployment_results_payload))

    @staticmethod
    def send_deployment_status(
            end_point_id, end_point_name, model_name, model_inference_url, model_status, message_center=None):
        if message_center is None:
            return
        deployment_status_topic = "model_ops/model_device/return_deployment_status"
        deployment_status_payload = {"end_point_id": end_point_id, "end_point_name": end_point_name,
                                     "model_name": model_name,
                                     "model_url": model_inference_url,
                                     "model_status": model_status,
                                     "timestamp": int(format(time.time_ns() / 1000.0, '.0f'))}
        logging.info(f"[Master] deployment_status_payload is sent to mlops: {deployment_status_payload}")

        message_center.send_message_json(deployment_status_topic, json.dumps(deployment_status_payload))

    @staticmethod
    def send_deployment_stages(end_point_id, model_name, model_id, model_inference_url,
                               model_stages_index, model_stages_title, model_stage_detail,
                               message_center=None):
        if message_center is None:
            return
        deployment_stages_topic = "model_ops/model_device/return_deployment_stages"
        deployment_stages_payload = {"model_name": model_name,
                                     "model_id": model_id,
                                     "model_url": model_inference_url,
                                     "end_point_id": end_point_id,
                                     "model_stage_index": model_stages_index,
                                     "model_stage_title": model_stages_title,
                                     "model_stage_detail": model_stage_detail,
                                     "timestamp": int(format(time.time_ns() / 1000.0, '.0f'))}

        message_center.send_message_json(deployment_stages_topic, json.dumps(deployment_stages_payload))

        logging.info(f"-------- Stages has been sent to mlops with stage {model_stages_index} and "
                     f"payload {deployment_stages_payload}")

    def send_deployment_start_request_to_edges(self, in_request_json=None):
        if in_request_json is not None:
            self.request_json = in_request_json

        # Iterate through replica_num_diff, both add and replace should be sent to the edge devices
        if "replica_num_diff" not in self.request_json or self.request_json["replica_num_diff"] is None:
            return []

        edge_id_list = []
        for device_id in self.request_json["replica_num_diff"].keys():
            edge_id_list.append(device_id)

        self.request_json["master_node_ip"] = GeneralConstants.get_ip_address(self.request_json)
        should_added_devices = []
        for edge_id in edge_id_list:
            if edge_id == self.edge_id:
                continue
            should_added_devices.append(edge_id)
            # send start deployment request to each device
            self.send_deployment_start_request_to_edge(edge_id, self.request_json)
        return should_added_devices

    def send_deployment_start_request_to_edge(self, edge_id, request_json):
        topic_start_deployment = "model_ops/model_device/start_deployment/{}".format(str(edge_id))
        logging.info("start_deployment: send topic " + topic_start_deployment + " to client...")
        self.message_center.send_message_json(topic_start_deployment, json.dumps(request_json))

    def send_deployment_delete_request_to_edges(self, payload, model_msg_object, message_center=None):
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
            if message_center is not None:
                message_center.send_message_json(topic_delete_deployment, payload)
            else:
                self.message_center.send_message_json(topic_delete_deployment, payload)

    def send_deployment_stop_request_to_edges(self, edge_id_list, payload):
        for edge_id in edge_id_list:
            topic_stop_deployment = "model_ops/model_device/stop_deployment/{}".format(str(self.edge_id))
            logging.info("stop_deployment: send topic " + topic_stop_deployment)
            self.message_center.send_message_json(topic_stop_deployment, payload)
