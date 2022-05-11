import json
import logging
import uuid

import paho.mqtt.client as mqtt


class MqttManager(object):
    MSG_MLOPS_CLIENT_STATUS_IDLE = "IDLE"
    MSG_MLOPS_CLIENT_STATUS_UPGRADING = "UPGRADING"
    MSG_MLOPS_CLIENT_STATUS_INITIALIZING = "INITIALIZING"
    MSG_MLOPS_CLIENT_STATUS_TRAINING = "TRAINING"
    MSG_MLOPS_CLIENT_STATUS_STOPPING = "STOPPING"
    MSG_MLOPS_CLIENT_STATUS_FINISHED = "FINISHED"

    def __init__(self, host, port, user, pwd, keepalive_time, id):
        self._host = host
        self._port = port
        self._client_id = str(id)
        self._listeners = dict()
        # use the run_id as the client ID
        self.mqtt_connection_id = mqtt.base62(uuid.uuid4().int, padding=22)
        self._client = mqtt.Client(client_id=self.mqtt_connection_id, clean_session=True)
        self._client.on_connect = self.on_connect
        self._client.on_publish = self.on_publish
        self._client.on_disconnect = self.on_disconnect
        self._client.on_message = self.on_message
        self._client.enable_logger()
        self._client.username_pw_set(user, pwd)

        _will_msg = {"ID": f"{self.mqtt_connection_id}", "stat": "Online"}
        self._client.will_set("W/topic", payload=json.dumps(_will_msg), qos=0, retain=True)
        self._client.connect(self._host, self._port, keepalive_time)

    def loop_forever(self):
        self._client.loop_forever()

    def send_message(self, topic, message):
        self._client.publish(topic, payload=message, qos=2)

    def send_message_json(self, topic, message):
        logging.info(message)
        self._client.publish(topic, payload=message, qos=2)

    def on_connect(self, client, userdata, flags, rc):
        # <sender>/<receiver>/<action>
        topic_start_train = "flserver_agent/" + str(self._client_id) + "/start_train"
        topic_stop_train = "flserver_agent/" + str(self._client_id) + "/stop_train"
        topic_client_status = "fl_client/mlops/" + str(self._client_id) + "/status"
        client.subscribe(topic_start_train)
        client.subscribe(topic_stop_train)
        client.subscribe(topic_client_status)
        logging.info("subscribe: " + topic_start_train)
        logging.info("subscribe: " + topic_stop_train)

    def on_message(self, client, userdata, msg):
        logging.info(f"on_message({msg.topic}, {str(msg.payload)})")
        _listener = self._listeners.get(msg.topic, None)
        if _listener is not None and callable(_listener):
            _listener(msg.topic, str(msg.payload, encoding="utf-8"))

    @staticmethod
    def on_publish(client, obj, mid):
        logging.info(f"on_publish mid={mid}")

    @staticmethod
    def on_disconnect(client, userdata, rc):
        logging.info(f"on_disconnect code={rc}")

    def add_message_listener(self, topic, listener):
        logging.info(f"add_message_listener({topic})")
        self._listeners[topic] = listener

    def remove_message_listener(self, topic):
        logging.info(f"remove_message_listener({topic})")
        del self._listeners[topic]
