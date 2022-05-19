# -*-coding:utf-8-*-
import json
import logging
import traceback
import uuid
from typing import List

import paho.mqtt.client as mqtt
import yaml

from .remote_storage import S3Storage
from ..base_com_manager import BaseCommunicationManager
from ..message import Message
from ..observer import Observer


class MqttS3StatusManager(BaseCommunicationManager):
    def __init__(self, config_path, s3_config_path, topic="fedml"):
        self._topic = "fedml_" + str(topic) + "_"
        self.s3_storage = S3Storage(s3_config_path)

        self._unacked_sub = list()
        self._observers: List[Observer] = []
        self._client_id = mqtt.base62(uuid.uuid4().int, padding=22)

        self.set_config_from_file(config_path)
        self.set_config_from_objects(config_path)
        # Construct a Client
        self.mqtt_connection_id = mqtt.base62(uuid.uuid4().int, padding=22)
        self._client = mqtt.Client(
            client_id=str(self.mqtt_connection_id), clean_session=True
        )
        self._client.enable_logger()

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.on_subscribe = self._on_subscribe
        # self._client.on_log = self.on_log

        self._client.username_pw_set(self.mqtt_user, self.mqtt_pwd)
        _will_msg = {"ID": f"{self.mqtt_connection_id}", "stat": "Online"}
        self._client.will_set(
            "W/topic", payload=json.dumps(_will_msg), qos=0, retain=True
        )

        self._client.connect(self.broker_host, self.broker_port, 180)

    def on_log(self, mqttc, obj, level, string):
        logging.info("mqtt_s3.on_log: " + string)

    def run_loop_forever(self):
        self._client.loop_forever()

    def __del__(self):
        self._client.loop_stop()
        self._client.disconnect()

    @property
    def client_id(self):
        return self._client_id

    @property
    def topic(self):
        return self._topic

    def _on_connect_impl(self, client, userdata, flags, rc):
        logging("mqtt_s3_status. connected")

    def _on_connect(self, client, userdata, flags, rc):
        try:
            self._on_connect_impl(client, userdata, flags, rc)
        except:
            traceback.print_exc()
            quit(0)

    @staticmethod
    def _on_disconnect(client, userdata, rc):
        logging.info(
            "mqtt_s3.on_disconnect: disconnection returned result %s, user data %s"
            % (str(rc), str(userdata))
        )

    def _on_subscribe(self, client, userdata, mid, granted_qos):
        logging.info("mqtt_s3.onSubscribe: mid = %s" % str(mid))
        self._unacked_sub.remove(mid)

    def subscribe(self, topic):
        self._client.subscribe(topic)

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def _notify(self, msg_obj):
        msg_params = Message()
        msg_params.init_from_json_object(msg_obj)
        msg_type = msg_params.get_type()
        logging.info("mqtt_s3.notify: msg type = %d" % msg_type)
        for observer in self._observers:
            observer.receive_message(msg_type, msg_params)

    def _on_message_impl(self, client, userdata, msg):
        logging.info("--------------------------")
        json_payload = str(msg.payload, encoding="utf-8")
        payload_obj = json.loads(json_payload)
        self._notify(payload_obj)

    def _on_message(self, client, userdata, msg):
        try:
            self._on_message_impl(client, userdata, msg)
        except:
            traceback.print_exc()
            quit(0)

    def send_message(self, msg: Message):
        topic = self._topic + str(msg.get_sender_id())
        payload = msg.get_params()
        self._client.publish(topic, payload=json.dumps(payload))

    def send_message_json(self, topic_name, json_message):
        self._client.publish(topic_name, payload=json_message)

    def handle_receive_message(self):
        self.run_loop_forever()

    def stop_receive_message(self):
        logging.info("mqtt_s3.stop_receive_message: stopping...")
        self._client.loop_stop()
        self._client.disconnect()

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
