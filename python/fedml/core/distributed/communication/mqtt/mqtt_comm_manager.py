# -*-coding:utf-8-*-
import logging
import uuid
from typing import List

import paho.mqtt.client as mqtt

from .mqtt_manager import MqttManager
from ..base_com_manager import BaseCommunicationManager
from ..message import Message
from ..observer import Observer


class MqttCommManager(BaseCommunicationManager):
    def __init__(self, host, port, topic="fedml", client_id=0, client_num=0):
        self._observers: List[Observer] = []
        self._topic = topic
        if client_id is None:
            self._client_id = mqtt.base62(uuid.uuid4().int, padding=22)
        else:
            self._client_id = client_id
        self.client_num = client_num

        self.mqtt_mgr = MqttManager(self.broker_host, self.broker_port, self.mqtt_user, self.mqtt_pwd,
                                    self.keepalive_time,
                                    self._client_id)
        self.mqtt_mgr.add_connected_listener(self.on_connected)
        self.mqtt_mgr.add_disconnected_listener(self.on_disconnected)
        self.mqtt_mgr.connect()

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
        """
        [server]
        sending message topic (publish): serverID_clientID
        receiving message topic (subscribe): clientID

        [client]
        sending message topic (publish): clientID
        receiving message topic (subscribe): serverID_clientID

        """
        logging.info("On_connected")

        # subscribe one topic
        if self.client_id == 0:
            # server
            for client_ID in range(1, self.client_num + 1):
                result, mid = mqtt_client_object.subscribe(self._topic + str(client_ID), 0)
                logging.info(result)
        else:
            # client
            result, mid = mqtt_client_object.subscribe(
                self._topic + str(0) + "_" + str(self.client_id), 0
            )
            logging.info(result)

    def _on_message(self, msg):
        msg.payload = str(msg.payload, encoding="utf-8")
        self._notify(str(msg.payload))

    def on_disconnected(self, mqtt_client_object):
        logging.info("On_disconnected")

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def _notify(self, msg):
        msg_params = Message()
        msg_params.init_from_json_string(str(msg))
        msg_type = msg_params.get_type()
        for observer in self._observers:
            observer.receive_message(msg_type, msg_params)

    def send_message(self, msg: Message):
        """
        [server]
        sending message topic (publish): serverID_clientID
        receiving message topic (subscribe): clientID

        [client]
        sending message topic (publish): clientID
        receiving message topic (subscribe): serverID_clientID

        """
        if self.client_id == 0:
            # server
            receiver_id = msg.get_receiver_id()
            topic = self._topic + str(0) + "_" + str(receiver_id)
            logging.info("topic = %s" % str(topic))
            payload = msg.to_json()
            self.mqtt_mgr.send_message(topic, payload)
            logging.info("sent")
        else:
            # client
            self.mqtt_mgr.send_message(
                self._topic + str(self.client_id), msg.to_json()
            )

    def handle_receive_message(self):
        pass

    def stop_receive_message(self):
        pass

