import argparse
import json
import logging
import traceback
import uuid

import paho.mqtt.client as mqtt
import paho.mqtt.publish as mqtt_publish
import time
from fedml.core.mlops.mlops_profiler_event import MLOpsProfilerEvent

import fedml

class MqttManager(object):
    def __init__(self, host, port, user, pwd, keepalive_time,
                 client_id, last_will_topic=None, last_will_msg=None,
                 clean_session=True, retain_msg=False):
        self._client = None
        self.mqtt_connection_id = None
        self._host = host
        self._port = port
        self.keepalive_time = keepalive_time
        self._client_id = str(client_id)
        self._listeners = dict()
        self._connected_listeners = list()
        self._disconnected_listeners = list()
        self._subscribed_listeners = list()
        self._published_listeners = list()
        self._passthrough_listeners = list()
        self.last_will_topic = last_will_topic
        self.last_will_msg = last_will_msg
        self.clean_session = clean_session
        self.retain_msg = retain_msg

        self.user = user
        self.pwd = pwd
        self.init_connect()

    def __del__(self):
        self._client.loop_stop()
        self._client.disconnect()
        self._connected_listeners.clear()
        self._disconnected_listeners.clear()
        self._subscribed_listeners.clear()
        self._published_listeners.clear()
        self._passthrough_listeners.clear()
        self.mqtt_connection_id = None
        self._client = None

    def init_connect(self):
        self.mqtt_connection_id = "{}_{}".format(self._client_id, "ID")
        self._client = mqtt.Client(client_id=self.mqtt_connection_id, clean_session=self.clean_session)
        self._client.connected_flag = False
        self._client.bad_conn_flag = False
        self._client.on_connect = self.on_connect
        self._client.on_publish = self.on_publish
        self._client.on_disconnect = self.on_disconnect
        self._client.on_message = self.on_message
        self._client.on_subscribe = self._on_subscribe
        # self._client.on_log = self._on_log
        self._client.disable_logger()
        self._client.username_pw_set(self.user, self.pwd)
        self._client._connect_timeout = 15
        # logging.info("MQTT Connection timeout: {}, client id {}.".format(
        #     self._client._connect_timeout, self.mqtt_connection_id))

    def connect(self):
        if self.last_will_topic is not None:
            if self.last_will_msg is None:
                self.last_will_msg = json.dumps({"ID": f"{self._client_id}", "status": "OFFLINE"})
            self._client.will_set(self.last_will_topic,
                                  payload=self.last_will_msg,
                                  qos=2, retain=self.retain_msg)
        self._client.connect(self._host, self._port, self.keepalive_time)

    def reconnect(self):
        try:
            self.init_connect()
            self._client.reconnect()
        except Exception as e:
            logging.info("Failed to reconnect to MQTT server: {}, client id {}".format(
                traceback.format_exc(), self.mqtt_connection_id))

    def disconnect(self):
        self._client.disconnect()

    def loop_start(self):
        self._client.loop_start()

    def loop_stop(self):
        self._client.loop_stop()
        self._client.unsubscribe

    def loop_forever(self):
        self._client.loop_forever(retry_first_connection=True)

    def send_message(self, topic, message, publish_single_message=False):
        # logging.info(
        #     f"FedMLDebug - Send: topic ({topic}), message ({message})"
        # )
        self.check_connection()

        mqtt_send_start_time = time.time()
        if publish_single_message:
            connection_id = "FEDML_SINGLE_CONN_{}_{}".format(self._client_id,
                                                             str(mqtt.base62(uuid.uuid4().int, padding=22)))
            mqtt_publish.single(topic, payload=message, qos=2,
                                hostname=self._host, port=self._port,
                                client_id=connection_id, retain=self.retain_msg,
                                auth={'username': self.user, 'password': self.pwd})
        else:
            ret_info = self._client.publish(topic, payload=message, qos=2, retain=self.retain_msg)
            return ret_info.is_published()
        MLOpsProfilerEvent.log_to_wandb({"Comm/send_delay_mqtt": time.time() - mqtt_send_start_time})
        return True

    def send_message_json(self, topic, message, publish_single_message=False):
        # logging.info(
        #     f"FedMLDebug - Send: topic ({topic}), message ({message})"
        # )
        self.check_connection()

        if publish_single_message:
            connection_id = "FEDML_SINGLE_CONN_{}_{}".format(self._client_id,
                                                             str(mqtt.base62(uuid.uuid4().int, padding=22)))
            mqtt_publish.single(topic, payload=message, qos=2,
                                hostname=self._host, port=self._port,
                                client_id=connection_id, retain=self.retain_msg,
                                auth={'username': self.user, 'password': self.pwd})
        else:
            ret_info = self._client.publish(topic, payload=message, qos=2, retain=self.retain_msg)
            return ret_info.is_published()
        return True

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.connected_flag = True
            client.bad_conn_flag = False
            # logging.info("MQTT Connection is OK, client id {}.".format(self.mqtt_connection_id))

            # Callback connected listeners
            self.callback_connected_listener(client)
        else:
            client.connected_flag = False
            client.bad_conn_flag = True

            if rc == 1:
                logging.info("MQTT Connection refused - incorrect protocol version, client id {}.".format(
                    self.mqtt_connection_id
                ))
            elif rc == 2:
                logging.info("MQTT Connection refused - invalid client identifier client id {}.".format(
                    self.mqtt_connection_id
                ))
            elif rc == 3:
                logging.info("MQTT Connection refused - server unavailable client id {}.".format(
                    self.mqtt_connection_id
                ))
            elif rc == 4:
                logging.info("MQTT Connection refused - bad username or password client id {}.".format(
                    self.mqtt_connection_id
                ))
            elif rc == 5:
                logging.info("MQTT Connection refused - not authorised client id {}.".format(
                    self.mqtt_connection_id
                ))
            else:
                logging.info("MQTT Connection failed - client id {}, return code {}.".format(
                    self.mqtt_connection_id, rc))

    def is_connected(self):
        return self._client.is_connected()

    def subscribe_will_set_msg(self, client):
        self.add_message_listener(self.last_will_topic, self.callback_will_set_msg)
        client.subscribe(self.last_will_topic, qos=2)

    def callback_will_set_msg(self, topic, payload):
        logging.info(f"MQTT client will be disconnected, id: {self._client_id}, topic: {topic}, payload: {payload}")

    def on_message(self, client, userdata, msg):
        # logging.info("on_message: msg.topic {}, msg.retain {}".format(msg.topic, msg.retain))

        if msg.retain:
            return

        message_handler_start_time = time.time()
        MLOpsProfilerEvent.log_to_wandb({"MessageReceiveTime": message_handler_start_time})
        for passthrough_listener in self._passthrough_listeners:
            passthrough_listener(msg)

        _listener = self._listeners.get(msg.topic, None)
        if _listener is not None and callable(_listener):
            payload_obj = json.loads(msg.payload)
            payload_obj["is_retain"] = msg.retain
            payload = json.dumps(payload_obj)
            _listener(msg.topic, payload)

        MLOpsProfilerEvent.log_to_wandb({"BusyTime": time.time() - message_handler_start_time})

    def on_publish(self, client, obj, mid):
        self.callback_published_listener(client)

    def on_disconnect(self, client, userdata, rc):
        client.connected_flag = False
        client.bad_conn_flag = True
        self.callback_disconnected_listener(client)

    def _on_subscribe(self, client, userdata, mid, granted_qos):
        self.callback_subscribed_listener(client)

    def _on_log(self, client, userdata, level, buf):
        logging.info("mqtt log {}, client id {}.".format(buf, self.mqtt_connection_id))

    def add_message_listener(self, topic, listener):
        self._listeners[topic] = listener

    def remove_message_listener(self, topic):
        try:
            del self._listeners[topic]
        except Exception as e:
            pass

    def add_message_passthrough_listener(self, listener):
        self.remove_message_passthrough_listener(listener)
        self._passthrough_listeners.append(listener)

    def remove_message_passthrough_listener(self, listener):
        try:
            self._passthrough_listeners.remove(listener)
        except Exception as e:
            pass

    def add_connected_listener(self, listener):
        self._connected_listeners.append(listener)

    def remove_connected_listener(self, listener):
        try:
            self._connected_listeners.remove(listener)
        except Exception as e:
            pass

    def callback_connected_listener(self, client):
        for listener in self._connected_listeners:
            if listener is not None and callable(listener):
                listener(client)

    def add_disconnected_listener(self, listener):
        self._disconnected_listeners.append(listener)

    def remove_disconnected_listener(self, listener):
        try:
            self._disconnected_listeners.remove(listener)
        except Exception as e:
            pass

    def callback_disconnected_listener(self, client):
        for listener in self._disconnected_listeners:
            if listener is not None and callable(listener):
                listener(client)

    def add_subscribed_listener(self, listener):
        self._subscribed_listeners.append(listener)

    def remove_subscribed_listener(self, listener):
        try:
            self._subscribed_listeners.remove(listener)
        except Exception as e:
            pass

    def callback_subscribed_listener(self, client):
        for listener in self._subscribed_listeners:
            if listener is not None and callable(listener):
                listener(client)

    def add_published_listener(self, listener):
        self._published_listeners.append(listener)

    def remove_published_listener(self, listener):
        try:
            self._published_listeners.remove(listener)
        except Exception as e:
            pass

    def callback_published_listener(self, client):
        for listener in self._published_listeners:
            if listener is not None and callable(listener):
                listener(client)

    def subscribe_msg(self, topic):
        self._client.subscribe(topic, qos=2)

    def unsubscribe_msg(self, topic):
        self._client.unsubscribe(topic)

    def check_connection(self):
        count = 0
        while not self._client.connected_flag and self._client.bad_conn_flag:
            if count >= 30:
                raise Exception("MQTT Connection timeout, please check your network connection!")
            logging.info("MQTT client id {}, waiting to connect to MQTT server...".format(self.mqtt_connection_id))
            time.sleep(1)
            count += 1

        if self._client.bad_conn_flag and not self.connected_flag:
            if not self._client.is_connected():
                logging.info("Failed to connect to MQTT server! {}".format(traceback.format_exc()))
                self._client.loop_stop()
                raise Exception("MQTT Connection failed, please check your network connection!")


global received_msg_count
received_msg_count = 0


def test_msg_callback(topic, payload):
    global received_msg_count
    received_msg_count += 1
    logging.info("Received the topic: {}, message: {}, count {}.".format(topic, payload, received_msg_count))


def test_last_will_callback(topic, payload):
    logging.info("Received the topic: {}, message: {}.".format(topic, payload))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--client_id", "-i", type=str, help="client id for mqtt.")
    parser.add_argument("--action", "-a", default="send", help="action, options: send or receive")
    parser.add_argument("--num", "-n", type=int, default=10, help="running number for sending or receiving")
    args = parser.parse_args()

    if args.action != "send" and args.action != "receive":
        logging.info("action must be the following options: send, receive.")
        exit(1)

    logging.getLogger().setLevel(logging.INFO)

    last_will_topic = "fedml/mqtt-test/lastwill"
    last_will_msg = {"ID": 1, "status": "OFFLINE"}
    mqtt_url = fedml._get_mqtt_service("release")
    mqtt_manager = MqttManager(mqtt_url, 1883, "admin", "test",
                               30, args.client_id,
                               last_will_topic=last_will_topic,
                               last_will_msg=json.dumps(last_will_msg))
    mqtt_manager.connect()
    mqtt_manager.loop_start()

    topic = "/fedml/mqtt-test/connect"
    if args.action == "receive":
        mqtt_manager.add_message_listener(topic, test_msg_callback)
        mqtt_manager.add_message_listener(last_will_topic, test_last_will_callback)
        mqtt_manager.add_message_listener("#", test_msg_callback)
        mqtt_manager.subscribe_msg(topic)
        mqtt_manager.subscribe_msg(last_will_topic)
        mqtt_manager.subscribe_msg("#")
    elif args.action == "send":
        for i in range(1, args.num):
            logging.info("send message {}, index {}.".format(topic, i))
            mqtt_manager.send_message(topic, "index {}".format(i))
            time.sleep(0.1)

    time.sleep(40000)
    mqtt_manager.loop_stop()
    mqtt_manager.disconnect()
