# -*-coding:utf-8-*-

import paho.mqtt.client as mqtt
import time
import uuid
import time
from typing import List

from fedml_core.distributed.communication import CommunicationManager, Observer


class MqttClient(CommunicationManager):
    def __init__(self, host, port, topic='hello', client_id=None):
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
        # subscribe topic
        result, mid = client.subscribe([("temperature", 0), ("humidity", 0)])
        self._unacked_sub.append(mid)
        print(result)
        print("Finish subscribe!")

    def _on_message(self, client, userdata, msg):
        print("Received message, topic:" + msg.topic + " payload:" + str(msg.payload))
        self._notify(msg.topic, str(msg.payload))

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

    def _notify(self, topic, msg):
        for observer in self._observers:
            observer.receive_message(topic, msg)

    def send(self, topic, msg):
        print("send(%s, %s)" % (topic, msg))
        self._client.publish(topic, payload=msg)


if __name__ == '__main__':
    class Obs(Observer):
        def receive_message(self, msg_type, msg_params) -> None:
            print("receive_message(%s,%s)" % (msg_type, msg_params))


    client = MqttClient("127.0.0.1", 1883)
    client.add_observer(Obs())
    time.sleep(3)
    print('client ID:%s' % client.client_id)
    client.send("hello", "Hello world!")
    client.send("temperature", "24.0")
    client.send("humidity", "65%")
    time.sleep(2)
    print("client, send Fin...")
