import json
import multiprocessing

from multiprocessing import Process
import os
import threading

import time
import uuid

from ....core.distributed.communication.mqtt.mqtt_manager import MqttManager
from fedml.computing.scheduler.model_scheduler.device_model_deployment import run_http_inference_with_curl_request

from ....core.mlops.mlops_metrics import MLOpsMetrics


class FedMLMqttInference:
    def __init__(self, request_json=None, agent_config=None, run_id=0, mqtt_mgr=None):
        self.run_id = run_id
        self.request_json = request_json
        self.run_inference_event_map = dict()
        self.run_inference_response_map = dict()
        self.mqtt_mgr = mqtt_mgr
        self.client_mqtt_mgr = None
        self.running_request_json = dict()
        self.endpoint_inference_runners = dict()
        self.mlops_metrics = None
        self.client_mqtt_lock = None
        self.client_mqtt_is_connected = False
        self.agent_config = agent_config

    def setup_listener_for_endpoint_inference_request(self, edge_id):
        inference_topic = f"fedml_model_master/fedml_model_worker/inference/{edge_id}"
        self.mqtt_mgr.add_message_listener(inference_topic, self.callback_endpoint_inference_request)
        self.mqtt_mgr.subscribe_msg(inference_topic)

    def remove_listener_for_endpoint_inference_request(self, edge_id):
        inference_topic = f"fedml_model_master/fedml_model_worker/inference/{edge_id}"
        self.mqtt_mgr.remove_message_listener(inference_topic)
        self.mqtt_mgr.unsubscribe_msg(inference_topic)

    def callback_endpoint_inference_request(self, topic, payload):
        request_json = json.loads(payload)
        run_id = request_json.get("endpoint_id", None)
        if run_id is None:
            return
        run_id = str(run_id)
        inference_request_id = request_json.get("inference_id", "0")
        client_runner = FedMLMqttInference(
            request_json=request_json, agent_config=self.agent_config, run_id=run_id
        )
        if self.endpoint_inference_runners.get(run_id, None) is None:
            self.endpoint_inference_runners[run_id] = dict()
        self.endpoint_inference_runners[run_id][inference_request_id] = Process(target=client_runner.inference_entry)
        self.endpoint_inference_runners[run_id][inference_request_id].start()

    def response_endpoint_inference(self, endpoint_id, inference_request_id, response):
        inference_response_topic = f"fedml_model_worker/fedml_model_master/inference/{endpoint_id}"
        payload_json = dict()
        payload_json["inference_id"] = inference_request_id
        payload_json["response"] = response

        self.client_mqtt_mgr.send_message_json(inference_response_topic, json.dumps(payload_json))
        self.release_client_mqtt_mgr()

    def inference_entry(self):
        endpoint_id = self.request_json.get("endpoint_id", "0")
        inference_request_id = self.request_json.get("inference_id", "0")
        inference_url = self.request_json.get("inference_url", "0")
        inference_input_list = self.request_json.get("input", "0")
        inference_output_list = self.request_json.get("output", "0")
        inference_type = self.request_json.get("inference_type", "0")

        self.setup_client_mqtt_mgr()

        inference_response = run_http_inference_with_curl_request(
            inference_url, inference_input_list, inference_output_list, inference_type=inference_type)

        self.response_endpoint_inference(endpoint_id, inference_request_id, inference_response)

    def run_mqtt_inference_with_request(self, edge_id, endpoint_id, inference_url, inference_input_list, inference_output_list,
                                        inference_type="default"):
        self.setup_client_mqtt_mgr()
        self.setup_listener_for_endpoint_inference_response(endpoint_id)

        str_endpoint_id = str(endpoint_id)
        inference_req_id = str(uuid.uuid4())
        if self.run_inference_event_map.get(str_endpoint_id) is None:
            self.run_inference_event_map[str_endpoint_id] = dict()
        self.run_inference_event_map[str_endpoint_id][inference_req_id] = multiprocessing.Event()
        self.run_inference_event_map[str_endpoint_id][inference_req_id].clear()

        self.send_mqtt_endpoint_inference_request(
            edge_id, endpoint_id, inference_req_id, inference_url,
            inference_input_list, inference_output_list, inference_type=inference_type)

        while True:
            if self.run_inference_event_map[str_endpoint_id][inference_req_id].is_set():
                return self.run_inference_response_map[str_endpoint_id][inference_req_id]

            time.sleep(0.05)

    def send_mqtt_endpoint_inference_request(
            self, edge_id, endpoint_id, inference_req_id, inference_url, inference_input_list,
            inference_output_list, inference_type="default"):
        inference_topic = f"fedml_model_master/fedml_model_worker/inference/{edge_id}"
        inference_request_json = dict()
        inference_request_json["endpoint_id"] = endpoint_id
        inference_request_json["inference_id"] = inference_req_id
        inference_request_json["inference_url"]  = inference_url
        inference_request_json["input"] = inference_input_list
        inference_request_json["output"] = inference_output_list
        inference_request_json["inference_type"] = inference_type

        self.client_mqtt_mgr.send_message_json(inference_topic, json.dumps(inference_request_json))

    def setup_listener_for_endpoint_inference_response(self, run_id):
        inference_response_topic = f"fedml_model_worker/fedml_model_master/inference/{run_id}"
        self.client_mqtt_mgr.add_message_listener(inference_response_topic, self.callback_endpoint_inference_response)
        self.client_mqtt_mgr.subscribe_msg(inference_response_topic)

    def remove_listener_for_endpoint_inference_response(self, run_id):
        inference_response_topic = f"fedml_model_worker/fedml_model_master/inference/{run_id}"
        self.client_mqtt_mgr.remove_message_listener(inference_response_topic)
        self.client_mqtt_mgr.unsubscribe_msg(inference_response_topic)

    def callback_endpoint_inference_response(self, topic, payload):
        str_endpoint_id = str(topic).split('/')[-1]
        if str_endpoint_id is None:
            return
        payload_json = json.loads(payload)
        inference_request_id = payload_json.get("inference_id", None)
        inference_response = payload_json.get("response", None)
        if inference_request_id is None:
            return

        if self.run_inference_response_map.get(str_endpoint_id) is None:
            self.run_inference_response_map[str_endpoint_id] = dict()
        self.run_inference_response_map[str_endpoint_id][inference_request_id] = inference_response
        if self.run_inference_event_map.get(str_endpoint_id) is None:
            self.run_inference_event_map[str_endpoint_id] = dict()
        if self.run_inference_event_map[str_endpoint_id].get(inference_request_id) is None:
            self.run_inference_event_map[str_endpoint_id][inference_request_id] = multiprocessing.Event()
        self.run_inference_event_map[str_endpoint_id][inference_request_id].set()

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
            "FedML_ModelInference_Metrics_{}_{}".format(str(os.getpid()), str(uuid.uuid4()))
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