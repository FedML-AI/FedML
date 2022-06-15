import json
import logging
import uuid

import paho.mqtt.client as mqtt


class MqttManager(object):
    MQTT_WILL_SET_TOPIC = "W/topic"

    def __init__(self, host, port, user, pwd, keepalive_time, client_id):
        self._host = host
        self._port = port
        self.keepalive_time = keepalive_time
        self._client_id = str(client_id)
        self._listeners = dict()
        self._connected_listeners = list()
        self._disconnected_listeners = list()
        self._subscribed_listeners = list()
        self._published_listeners = list()

        self.mqtt_connection_id = mqtt.base62(uuid.uuid4().int, padding=22)
        self._client = mqtt.Client(client_id=self.mqtt_connection_id, clean_session=True)
        self._client.on_connect = self.on_connect
        self._client.on_publish = self.on_publish
        self._client.on_disconnect = self.on_disconnect
        self._client.on_message = self.on_message
        self._client.on_subscribe = self._on_subscribe
        self._client.enable_logger()
        self._client.username_pw_set(user, pwd)

    def connect(self):
        _will_msg = {"ID": f"{self._client_id}", "stat": "Online"}
        self._client.will_set(MqttManager.MQTT_WILL_SET_TOPIC, payload=json.dumps(_will_msg), qos=0, retain=True)
        self._client.connect(self._host, self._port, self.keepalive_time)

    def disconnect(self):
        self._client.disconnect()

    def loop_start(self):
        self._client.loop_start()

    def loop_stop(self):
        self._client.loop_stop()

    def loop_forever(self):
        self._client.loop_forever()

    def send_message(self, topic, message):
        self._client.publish(topic, payload=message, qos=2)

    def send_message_json(self, topic, message):
        logging.info(message)
        self._client.publish(topic, payload=message, qos=2)

    def on_connect(self, client, userdata, flags, rc):
        # Subscribe will set message for reporting whether the connection is alive.
        self.subscribe_will_set_msg(client)

        # Callback connected listeners
        self.callback_connected_listener(client)

    def subscribe_will_set_msg(self, client):
        self.add_message_listener(MqttManager.MQTT_WILL_SET_TOPIC, self.callback_will_set_msg)
        client.subscribe(MqttManager.MQTT_WILL_SET_TOPIC)

    def callback_will_set_msg(self, topic, payload):
        logging.info(f"MQTT client will be disconnected, id: {self._client_id}, topic: {topic}, payload: {payload}")

    def on_message(self, client, userdata, msg):
        logging.info(f"on_message({msg.topic}, {str(msg.payload)})")
        _listener = self._listeners.get(msg.topic, None)
        if _listener is not None and callable(_listener):
            _listener(msg.topic, str(msg.payload, encoding="utf-8"))

    def on_publish(self, client, obj, mid):
        logging.info(f"on_publish mid={mid}")
        self.callback_published_listener(client)

    def on_disconnect(self, client, userdata, rc):
        logging.info(f"on_disconnect code={rc}")
        self.callback_disconnected_listener(client)

    def _on_subscribe(self, client, userdata, mid, granted_qos):
        logging.info(f"onSubscribe: mid = {str(mid)}")
        self.callback_subscribed_listener(client)

    def add_message_listener(self, topic, listener):
        logging.info(f"add_message_listener({topic})")
        self._listeners[topic] = listener

    def remove_message_listener(self, topic):
        logging.info(f"remove_message_listener({topic})")
        del self._listeners[topic]

    def add_connected_listener(self, listener):
        self._connected_listeners.append(listener)

    def remove_connected_listener(self, listener):
        self._connected_listeners.remove(listener)

    def callback_connected_listener(self, client):
        logging.info("callback_connected_listener: {}".format(len(self._connected_listeners)))
        for listener in self._connected_listeners:
            if listener is not None and callable(listener):
                listener(client)

    def add_disconnected_listener(self, listener):
        self._disconnected_listeners.append(listener)

    def remove_disconnected_listener(self, listener):
        self._disconnected_listeners.remove(listener)

    def callback_disconnected_listener(self, client):
        for listener in self._disconnected_listeners:
            if listener is not None and callable(listener):
                listener(client)

    def add_subscribed_listener(self, listener):
        self._subscribed_listeners.append(listener)

    def remove_subscribed_listener(self, listener):
        self._subscribed_listeners.remove(listener)

    def callback_subscribed_listener(self, client):
        for listener in self._subscribed_listeners:
            if listener is not None and callable(listener):
                listener(client)

    def add_published_listener(self, listener):
        self._published_listeners.append(listener)

    def remove_published_listener(self, listener):
        self._published_listeners.remove(listener)

    def callback_published_listener(self, client):
        for listener in self._published_listeners:
            if listener is not None and callable(listener):
                listener(client)
