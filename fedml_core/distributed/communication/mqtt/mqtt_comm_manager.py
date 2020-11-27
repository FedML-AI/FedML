# -*-coding:utf-8-*-
import logging
import time
import uuid
from typing import List

import paho.mqtt.client as mqtt

from ..base_com_manager import BaseCommunicationManager
from ..message import Message
from ..observer import Observer


class MqttCommManager(BaseCommunicationManager):
    def __init__(self, host, port, topic='fedml', client_id=0, client_num=0):
        self._unacked_sub = list()
        self._observers: List[Observer] = []
        self._topic = topic
        if client_id is None:
            self._client_id = mqtt.base62(uuid.uuid4().int, padding=22)
        else:
            self._client_id = client_id
        self.client_num = client_num
        # Construct a Client
        self._client = mqtt.Client(client_id=str(self._client_id))
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.on_subscribe = self._on_subscribe
        # connect broker,connect() or connect_async()
        self._client.connect(host, port, 60)
        self._client.loop_start()
        # self._client.loop_forever()

    def __del__(self):
        self._client.loop_stop()
        self._client.disconnect()

    @property
    def client_id(self):
        return self._client_id

    @property
    def topic(self):
        return self._topic

    def _on_connect(self, client, userdata, flags, rc):
        """
            [server]
            sending message topic (publish): serverID_clientID
            receiving message topic (subscribe): clientID

            [client]
            sending message topic (publish): clientID
            receiving message topic (subscribe): serverID_clientID

        """
        print("Connection returned with result code:" + str(rc))
        # subscribe one topic
        if self.client_id == 0:
            # server
            for client_ID in range(1, self.client_num+1):
                result, mid = self._client.subscribe(self._topic + str(client_ID), 0)
                self._unacked_sub.append(mid)
                print(result)
        else:
            # client
            result, mid = self._client.subscribe(self._topic + str(0) + "_" + str(self.client_id), 0)
            self._unacked_sub.append(mid)
            print(result)

    def _on_message(self, client, userdata, msg):
        msg.payload = str(msg.payload, encoding='utf-8')
        # print("_on_message: " + str(msg.payload))
        self._notify(str(msg.payload))

    @staticmethod
    def _on_disconnect(client, userdata, rc):
        print("Disconnection returned result:" + str(rc))

    def _on_subscribe(self, client, userdata, mid, granted_qos):
        print("onSubscribe :" + str(mid))
        self._unacked_sub.remove(mid)

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def _notify(self, msg):
        # print("_notify: " + msg)
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
            self._client.publish(topic, payload=payload)
            logging.info("sent")
        else:
            # client
            self._client.publish(self._topic + str(self.client_id), payload=msg.to_json())

    def handle_receive_message(self):
        pass

    def stop_receive_message(self):
        pass


if __name__ == '__main__':
    class Obs(Observer):
        def receive_message(self, msg_type, msg_params) -> None:
            print("receive_message(%s, %s)" % (msg_type, msg_params.to_string()))
    
    client = MqttCommManager("81.71.1.31", 1883)
    client.add_observer(Obs())
    time.sleep(3)
    print('client ID:%s' % client.client_id)

    message = Message(0, 1, 2)
    message.add_params("key1", 1)
    client.send_message(message)

    time.sleep(10)
    print("client, send Fin...")
