# -*-coding:utf-8-*-
import json
import logging
import threading
import time
import traceback
import uuid
from typing import List

import paho.mqtt.client as mqtt
import yaml

from ..mqtt.mqtt_manager import MqttManager
from .remote_storage import S3Storage
from ..base_com_manager import BaseCommunicationManager
from ..message import Message
from ..observer import Observer


class MqttS3StatusManager(BaseCommunicationManager):
    def __init__(self, config_path, s3_config_path, topic="fedml"):
        self.mqtt_pwd = None
        self.mqtt_user = None
        self.broker_port = None
        self.broker_host = None
        self.keepalive_time = 180
        self.mqtt_mgr_is_connected = False
        self.mqtt_mgr_lock = threading.Lock()

        self._topic = "fedml_" + str(topic) + "_"
        self.s3_storage = S3Storage(s3_config_path)

        self._observers: List[Observer] = []
        self._client_id = mqtt.base62(uuid.uuid4().int, padding=22)

        self.set_config_from_file(config_path)
        self.set_config_from_objects(config_path)

        self.mqtt_mgr = MqttManager(self.broker_host, self.broker_port, self.mqtt_user, self.mqtt_pwd,
                                    self.keepalive_time,
                                    self._client_id)
        self.mqtt_mgr.add_connected_listener(self.on_connected)
        self.mqtt_mgr.add_disconnected_listener(self.on_disconnected)
        self.mqtt_mgr.connect()
        self.mqtt_mgr.loop_start()

    def wait_connected(self):
        while True:
            self.mqtt_mgr_lock.acquire()
            if self.mqtt_mgr_is_connected is True:
                self.mqtt_mgr_lock.release()
                break
            self.mqtt_mgr_lock.release()
            time.sleep(1)

    def run_loop_forever(self):
        self.mqtt_mgr.loop_forever()

    def __del__(self):
        self.mqtt_mgr.loop_stop()
        self.mqtt_mgr.disconnect()

    @property
    def client_id(self):
        return self._client_id

    @property
    def topic(self):
        return self._topic

    def on_connected(self, mqtt_client_object):
        self.mqtt_mgr_lock.acquire()
        self.mqtt_mgr_is_connected = True
        self.mqtt_mgr_lock.release()
        logging.info("mqtt_s3_status. on_connected")

    def on_disconnected(self, mqtt_client_object):
        self.mqtt_mgr_lock.acquire()
        self.mqtt_mgr_is_connected = False
        self.mqtt_mgr_lock.release()
        logging.info(
            "mqtt_s3_status.on_disconnected"
        )

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def _notify(self, msg_obj):
        msg_params = Message()
        msg_params.init_from_json_object(msg_obj)
        msg_type = msg_params.get_type()
        logging.info("mqtt_s3_status.notify: msg type = %d" % msg_type)
        for observer in self._observers:
            observer.receive_message(msg_type, msg_params)

    def _on_message_impl(self, client, userdata, msg):
        json_payload = str(msg.payload, encoding="utf-8")
        payload_obj = json.loads(json_payload)
        self._notify(payload_obj)

    def _on_message(self, client, userdata, msg):
        try:
            self._on_message_impl(client, userdata, msg)
        except Exception as e:
            logging.error("mqtt_s3_status exception: {}".format(traceback.format_exc()))

    def send_message(self, msg: Message):
        self.wait_connected()
        topic = self._topic + str(msg.get_sender_id())
        payload = msg.get_params()
        self.mqtt_mgr.send_message(topic, json.dumps(payload))

    def send_message_json(self, topic_name, json_message):
        self.wait_connected()
        self.mqtt_mgr.send_message_json(topic_name, json_message)

    def handle_receive_message(self):
        self.run_loop_forever()

    def stop_receive_message(self):
        logging.info("mqtt_s3.stop_receive_message: stopping...")
        self.mqtt_mgr.loop_stop()
        self.mqtt_mgr.disconnect()

    def set_config_from_file(self, config_file_path):
        try:
            with open(config_file_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                self.broker_host = config["BROKER_HOST"]
                self.broker_port = config["BROKER_PORT"]
                self.mqtt_user = None
                self.mqtt_pwd = None
                if "MQTT_USER" in config:
                    self.mqtt_user = config["MQTT_USER"]
                if "MQTT_PWD" in config:
                    self.mqtt_pwd = config["MQTT_PWD"]
        except Exception as e:
            pass

    def set_config_from_objects(self, mqtt_config):
        self.broker_host = mqtt_config["BROKER_HOST"]
        self.broker_port = mqtt_config["BROKER_PORT"]
        self.mqtt_user = None
        self.mqtt_pwd = None
        if "MQTT_USER" in mqtt_config:
            self.mqtt_user = mqtt_config["MQTT_USER"]
        if "MQTT_PWD" in mqtt_config:
            self.mqtt_pwd = mqtt_config["MQTT_PWD"]
