# -*-coding:utf-8-*-

import time
import uuid
from typing import List

import paho.mqtt.client as mqtt

from fedml_core.distributed.communication.base_com_manager import BaseCommunicationManager
from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.communication.observer import Observer


class MqttCommManager(BaseCommunicationManager):
    def __init__(self, host, port, topic='fedml', client_id=None):
        self._unacked_sub = list()
        self._observers: List[Observer] = []
        self._topic = topic
        if client_id is None:
            self._client_id = mqtt.base62(uuid.uuid4().int, padding=22)
        else:
            self._client_id = client_id
        # Construct a Client
        self._client = mqtt.Client(client_id=self._client_id)
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
        print("Connection returned with result code:" + str(rc))
        # subscribe one topic
        result, mid = self._client.subscribe(self._topic, 0)
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
        # print(msg.to_string())
        self._client.publish(self._topic, payload=msg.to_json())

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
