import json
import uuid

import paho.mqtt.client as mqtt

import fedml

BROKER_HOST = "mqtt.fedml.ai"
BROKER_PORT = 1883
MQTT_KEEPALIVE = 60
EDGE_ID = "EDGE-%s" % uuid.uuid4().hex
MQTT_USER = "admin"
MQTT_PWD = "password"


class EdgeCommunicator(object):
    def __init__(self, host=BROKER_HOST, port=BROKER_PORT, client_id=EDGE_ID):
        self._host = host
        self._port = port
        self._client_id = client_id
        self._listeners = dict()
        # 建立连接
        self._client = mqtt.Client(client_id=client_id, clean_session=True)
        # 检查hostname的cert认证
        # self._client.tls_insecure_set(True)
        # 设置认证文件
        # self._client.tls_set(trust)
        self._client.on_connect = self.on_connect
        self._client.on_publish = self.on_publish
        self._client.on_disconnect = self.on_disconnect
        self._client.on_message = self.on_message
        self._client.enable_logger()
        self._client.username_pw_set(MQTT_USER, MQTT_PWD)
        # 遗言消息定义
        _will_msg = {"ID": f"{client_id}", "stat": "Online"}
        # 遗言消息 一旦连接到MQTT服务器，遗言消息就会被服务器托管，本客户端凡是非正常断开连接
        # 服务器就会将本遗言发送给订阅该遗言消息的客户端告知对方本客户端离线；
        self._client.will_set(
            "W/topic", payload=json.dumps(_will_msg), qos=0, retain=True
        )
        self._client.connect(self._host, self._port, MQTT_KEEPALIVE)
        self._client.loop_start()

    def send_message(self, topic, message):
        self._client.publish(topic, payload=message, qos=0)

    def on_connect(self, client, userdata, flags, rc):
        fedml.logger.info(f"Connected with result code {rc}")
        client.subscribe("EDGE/#")
        client.subscribe(f"EDGE/{self._client_id}")

    def on_message(self, client, userdata, msg):
        fedml.logger.info(f"on_message({msg.topic}, {str(msg.payload)})")
        _listener = self._listeners.get(msg.topic, None)
        if _listener is not None and callable(_listener):
            _listener(msg.topic, str(msg.payload))

    @staticmethod
    def on_publish(client, obj, mid):
        fedml.logger.info(f"on_publish mid={mid}")

    @staticmethod
    def on_disconnect(client, userdata, rc):
        fedml.logger.info(f"on_disconnect code={rc}")

    def add_message_listener(self, topic, listener):
        fedml.logger.info(f"add_message_listener({topic})")
        self._listeners[topic] = listener

    def remove_message_listener(self, topic):
        fedml.logger.info(f"remove_message_listener({topic})")
        del self._listeners[topic]
