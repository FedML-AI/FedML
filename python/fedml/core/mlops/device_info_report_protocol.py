import json
import multiprocessing
import os
import threading
import time
import uuid

from fedml.core.distributed.communication.mqtt.mqtt_manager import MqttManager


class FedMLDeviceInfoReportProtocol:
    def __init__(self, request_json=None, agent_config=None, run_id=0, mqtt_mgr=None):
        self.run_id = run_id
        self.request_json = request_json
        self.run_device_info_event_map = dict()
        self.run_device_info_response_map = dict()
        self.client_mqtt_mgr = mqtt_mgr
        self.running_request_json = dict()
        self.client_mqtt_lock = None
        self.client_mqtt_is_connected = False
        self.agent_config = agent_config

    def get_device_info(self, run_id, edge_id, server_id):
        str_run_id = str(run_id)
        str_edge_id = str(edge_id)
        self.setup_client_mqtt_mgr()
        self.setup_listener_for_device_info_response(server_id)

        if self.run_device_info_event_map.get(str_edge_id) is None:
            self.run_device_info_event_map[str_edge_id] = multiprocessing.Event()
        self.run_device_info_event_map[str_edge_id].clear()

        self.request_device_info(run_id, edge_id, server_id)

        sleep_time = 0.05
        allowed_timeout = 15
        total_timeout_counter = 0
        while True:
            if self.run_device_info_event_map[str_edge_id].is_set():
                return self.run_device_info_response_map[str_edge_id]

            total_timeout_counter += sleep_time
            time.sleep(sleep_time)

            if total_timeout_counter >= allowed_timeout:
                break

        return None

    def setup_listener_for_device_info_response(self, server_id):
        response_topic = f"client/server/response_device_info/{server_id}"
        self.client_mqtt_mgr.add_message_listener(response_topic, self.callback_device_info_response)
        self.client_mqtt_mgr.subscribe_msg(response_topic)

    def remove_listener_for_device_info_response(self, server_id):
        response_topic = f"client/server/response_device_info/{server_id}"
        self.client_mqtt_mgr.remove_message_listener(response_topic)
        self.client_mqtt_mgr.unsubscribe_msg(response_topic)

    def callback_device_info_response(self, topic, payload):
        # Parse payload
        payload_json = json.loads(payload)
        run_id = payload_json.get("run_id", 0)
        master_device_id = payload_json.get("master_device_id", 0)
        slave_device_id = payload_json.get("slave_device_id", 0)
        edge_id = payload_json.get("edge_id", 0)
        device_info = payload_json.get("edge_info", 0)
        device_info["master_device_id"] = master_device_id
        device_info["slave_device_id"] = slave_device_id
        run_id_str = str(run_id)
        edge_id_str = str(edge_id)

        # Save the device info
        self.run_device_info_response_map[edge_id_str] = device_info
        if self.run_device_info_event_map.get(edge_id_str) is None:
            self.run_device_info_event_map[edge_id_str] = multiprocessing.Event()
        self.run_device_info_event_map[edge_id_str].set()

    def on_client_mqtt_disconnected(self, mqtt_client_object):
        if self.client_mqtt_lock is None:
            self.client_mqtt_lock = threading.Lock()

        self.client_mqtt_lock.acquire()
        self.client_mqtt_is_connected = False
        self.client_mqtt_lock.release()

    def on_client_mqtt_connected(self, mqtt_client_object):
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
            "FedML_DeviceInfoReporter_Metrics_{}_{}".format(str(os.getpid()), str(uuid.uuid4()))
        )

        self.client_mqtt_mgr.add_connected_listener(self.on_client_mqtt_connected)
        self.client_mqtt_mgr.add_disconnected_listener(self.on_client_mqtt_disconnected)
        self.client_mqtt_mgr.connect()
        self.client_mqtt_mgr.loop_start()

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

    def request_device_info(self, run_id, edge_id, server_id):
        topic_request_device_info = "server/client/request_device_info/" + str(edge_id)
        payload = {"server_id": server_id, "run_id": run_id, "need_running_process_list": True}
        self.client_mqtt_mgr.send_message(topic_request_device_info, json.dumps(payload))