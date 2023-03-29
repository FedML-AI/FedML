import { BaseCommunicationManager } from './base_com_manager';
import { Message } from './message';
import { Observer } from './observer';
import { CommunicationConstants } from './constants';

import * as mqtt from 'mqtt/dist/mqtt.min';

export class MqttS3MultiClientsCommManager implements BaseCommunicationManager {
  config_path;
  s3_config_path;
  topic;
  client_rank;
  client_num;
  args;
  broker_port;
  broker_host;
  mqtt_pwd;
  mqtt_user;
  keepalive_time = 180;
  client_id_list;
  _topic;
  s3_storage;
  server_id: number;
  client_real_ids;
  group_server_id_list;
  edge_id;
  observers: Array<Observer>;
  _client_id: string;
  client_active_list: Object;
  top_active_msg;
  topic_last_will_msg;
  last_will_msg;
  mqtt_mgr;
  is_connected;
  options: Object;
  _listeners;
  _connected_listeners;
  _subscribed_listeners;
  _published_listeners;
  constructor(
    config_path,
    s3_config_path,
    topic = 'fedml',
    client_rank = 0,
    client_num = 0,
    args = null,
  ) {
    this.broker_port = null;
    this.broker_host = null;
    this.mqtt_pwd = null;
    this.keepalive_time = 180;
    this.client_id_list = args.client_id_list;
    this._topic = 'fedml_' + topic + '_';
    // this.s3_storage = new S3Storage(s3_config_path);
    this._connected_listeners = [];
    this._subscribed_listeners = [];
    this._published_listeners = [];
    this.client_real_ids = [];
    if (args.client_id_list != null) {
      console.log('MqttS3CommManager args client_id_list: ', args.client_id_list);
      this.client_real_ids = args.client_id_list;
    }
    this.group_server_id_list = null;
    if (args.group_server_id_list != null) {
      this.group_server_id_list = args.group_server_id_list;
    }

    if (args.server_id != null) {
      this.server_id = args.server_id;
    } else {
      this.server_id = 0;
    }
    if (args.client_id != null) {
      this.edge_id = args.client_id;
    }
    this.observers = [];
    if (client_rank == null) {
      this._client_id = 'mqttjs_' + Math.random().toString(16).substr(2, 8);
    } else {
      this._client_id = String(client_rank);
    }
    this.client_num = client_num;
    console.log('mqtt_s3.init: client_num = ', client_num);
    this.set_config_from_objects(config_path);
    this.client_active_list = {};
    this.last_will_msg = JSON.stringify({
      ID: this.edge_id,
      status: CommunicationConstants.MSG_CLIENT_STATUS_OFFLINE,
    });
    this.top_active_msg = CommunicationConstants.CLIENT_TOP_ACTIVE_MSG;
    this.topic_last_will_msg = CommunicationConstants.CLIENT_TOP_LAST_WILL_MSG;
    this.last_will_msg = JSON.stringify({
      ID: this.edge_id,
      status: CommunicationConstants.MSG_CLIENT_STATUS_OFFLINE,
    });
    this.options = {
      username: this.mqtt_user,
      password: this.mqtt_pwd,
      cleanSession: false,
      keepAlive: 180,
      clientId: 'mqttjs_' + Math.random().toString(16).substr(2, 8),
      connectTimeout: 4000,
    };
    const connectUrl = 'wss://open-dev.fedml.ai/mqtt';
    this.mqtt_mgr = mqtt.connect(connectUrl, this.options);
    this.mqtt_mgr.on('connect', () => {
      console.log('Connected');
      this.callback_connected_listener(this.mqtt_mgr);
      this.on_connect();
    });
    this.mqtt_mgr.on('message', (topic, message) => {
      // message is Buffer
      // this.on_message(message);
      const msgData = message.toString();
      const jsonData = JSON.parse(msgData);
      console.log('mqtt-jsonData ', jsonData);
      // this.mqtt_mgr.add_message_passthrough_listener(this.on_message(message));
      const s3_key_str = jsonData[Message.MSG_ARG_KEY_MODEL_PARAMS];
      this._notify(jsonData);
      console.log(message.toString());
    });
    // this.mqtt_mgr.on_disconnect = self.on_disconnect;

    this.is_connected = false;
  }

  // on_message(msg) {
  //   const msgData = msg.toString();
  //   const jsonData = JSON.parse(msgData);
  //   this.mqtt_mgr.add_message_passthrough_listener(this.on_message(msg));
  //   const s3_key_str = jsonData[Message.MSG_ARG_KEY_MODEL_PARAMS];
  //   this._notify(jsonData);
  // }

  callback_published_listener(client) {
    this._published_listeners.forEach((listener) => {
      listener(client);
    });
  }

  callback_connected_listener(client) {
    this._connected_listeners.forEach((listener) => {
      console.log('listener ', listener);
      listener(client);
    });
  }

  callback_subscribed_listener(client) {
    this._subscribed_listeners.forEach((listener) => {
      listener(client);
    });
  }

  _on_subscribe(client) {
    this.callback_subscribed_listener(client);
  }

  on_connect() {
    if (this.is_connected) {
      return;
    }

    const real_topic = this._topic + String(this.server_id) + '_' + String(this.client_real_ids[0]);
    console.log('real_topic ', real_topic);
    this.mqtt_mgr.subscribe(real_topic, 0);
    this._notify_connection_ready();

    this.is_connected = true;
  }

  on_disconnect() {
    this.is_connected = false;
  }

  get_client_id() {
    return this._client_id;
  }
  get_topic() {
    return this._topic;
  }

  _notify_connection_ready() {
    const msg_params = new Message();
    const msg_type = CommunicationConstants.MSG_TYPE_CONNECTION_IS_READY;
    this.observers.forEach((item) => {
      console.log('!!!!!!item ', item);
      item.receive_message(msg_type, msg_params);
    });
  }

  _notify(msg_obj) {
    const msg_params = new Message();
    msg_params.init_from_json_object(msg_obj);
    const msg_type = msg_params.get_type();
    this.observers.forEach((item) => {
      item.receive_message(msg_type, msg_params);
    });
  }

  set_config_from_objects(mqtt_config) {
    this.broker_host = mqtt_config.BROKER_HOST;
    this.broker_port = mqtt_config.BROKER_PORT;
    this.mqtt_user = null;
    this.mqtt_pwd = null;
    if (mqtt_config.MQTT_USER != null) {
      this.mqtt_user = mqtt_config.MQTT_USER;
    }
    if (mqtt_config.MQTT_PWD != null) {
      this.mqtt_pwd = mqtt_config.MQTT_PWD;
    }
  }
  send_message(msg: Message) {
    const sender_id = msg.get_sender_id();
    const receiver_id = msg.get_receiver_id();
    const topic = 'fedml_0_0_1';
    const uuid = this.guid();
    const message_key = topic + '_' + uuid;
    const payload = msg.get_params();
    payload.msg_type = parseInt(payload.msg_type);
    console.log('!!!!!!topic ', topic);
    console.log('!!!!!!payload ', JSON.stringify(payload));
    // const model_params_obj = payload[Message.MSG_ARG_KEY_MODEL_PARAMS];
    this.mqtt_mgr.publish(topic, JSON.stringify(payload), function (error) {
      if (error) {
        console.log(error);
      } else {
        console.log('Published!!!!!!!');
      }
    });
  }
  guid() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
      const r = (Math.random() * 16) | 0,
        v = c == 'x' ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }
  add_observer(observer: Observer) {
    this.observers.push(observer);
    console.log('!!!!!!!!!!!', this.observers);
  }
  remove_observer(observer: Observer) {
    throw new Error('Method not implemented.');
  }
  handle_receive_message() {
    console.log('ListenStart');
    this.mqtt_mgr.handleMessage();
  }
  stop_receive_message() {
    console.log('stop receive message');
  }
}

export default MqttS3MultiClientsCommManager;
