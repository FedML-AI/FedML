# -*-coding:utf-8-*-

import paho.mqtt.client as mqtt
import time


class MqttClient(object):
    def __init__(self, host, port, on_message=None, topic='hello'):
        self._unacked_sub = list()
        self._topic = topic
        # Construct a Client
        self._client = mqtt.Client()
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        if on_message is not None:
            self._client.on_message = on_message
        else:
            self._client.on_message = self._on_message
        self._client.on_subscribe = self._on_subscribe
        # connect broker,connect() or connect_async()
        self._client.connect(host, port, 60)
        self._client.loop_start()
        # self._client.loop_forever()

    def __del__(self):
        self._client.loop_stop()
        self._client.disconnect()

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

    @staticmethod
    def _on_message(client, userdata, msg):
        print("Received message, topic:" + msg.topic + "payload:" + str(msg.payload))

    @staticmethod
    def _on_disconnect(client, userdata, rc):
        print("Disconnection returned result:" + str(rc))

    def _on_subscribe(self, client, userdata, mid, granted_qos):
        print("onSubscribe :" + str(mid))
        self._unacked_sub.remove(mid)

    def send(self, topic, msg):
        print("send(%s, %s)" % (topic, msg))
        self._client.publish(topic, payload=msg)


if __name__ == '__main__':
    client = MqttClient("127.0.0.1", 1883)
    time.sleep(10)
    client.send("hello", "Hello world!")
    client.send("temperature", "24.0")
    client.send("humidity", "65%")
    time.sleep(5)
    print("client, send Fin...")
