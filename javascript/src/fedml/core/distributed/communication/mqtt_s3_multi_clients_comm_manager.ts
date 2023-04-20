import type { Rank } from '@tensorflow/tfjs'
import { Tensor, transpose } from '@tensorflow/tfjs'
import { v4 as uuidv4 } from 'uuid'
import * as mqtt from 'mqtt/dist/mqtt.min'
import type { BaseCommunicationManager } from './base_com_manager'
import { Message } from './message'
import type { Observer } from './observer'
import { CommunicationConstants } from './constants'
import { S3Storage } from './remote_storage'

const mqttUrl = 'wss://open.fedml.ai/mqtt'

export class MqttS3MultiClientsCommManager implements BaseCommunicationManager {
  // mqtt config
  config_path
  // s3 config
  s3_config_path
  topic: any
  client_rank: any
  client_num
  args: any
  broker_port: null
  broker_host: null
  mqtt_pwd?: string | null
  mqtt_user?: string | null
  keepalive_time = 180
  client_id_list: null
  _topic
  s3_storage
  server_id: number
  client_real_ids: any[]
  group_server_id_list: null
  edge_id: any
  observers: Array<Observer>
  _client_id: string
  client_active_list: Object
  top_active_msg
  topic_last_will_msg
  last_will_msg
  mqtt_mgr
  is_connected
  options: Object
  topic_in
  _listeners: any
  _connected_listeners: Array<Function>
  _subscribed_listeners: Array<Function>
  _published_listeners: Array<Function>
  _passthrough_listeners: any

  constructor(
    config_path: { BROKER_HOST: any; MQTT_PWD: any; BROKER_PORT: any; MQTT_KEEPALIVE: any; MQTT_USER: any },
    s3_config_path: { CN_S3_SAK: any; CN_REGION_NAME: any; CN_S3_AKI: any; BUCKET_NAME: any },
    topic_in = 'fedml',
    client_rank = 0,
    client_num = 0,
    args: { client_id_list: null; group_server_id_list: null; server_id: number; client_id: null },
  ) {
    this.config_path = config_path
    this.s3_config_path = s3_config_path
    this.broker_port = null
    this.broker_host = null
    this.mqtt_pwd = null
    this.keepalive_time = 180
    this.topic_in = topic_in
    this.client_id_list = args.client_id_list
    this._topic = `fedml_${this.topic_in}_`
    this.s3_storage = new S3Storage(this.s3_config_path)
    this._connected_listeners = []
    this._subscribed_listeners = []
    this._published_listeners = []
    this.client_real_ids = []
    if (args.client_id_list != null) {
      console.log(
        'MqttS3CommManager args client_id_list: ',
        args.client_id_list,
      )
      this.client_real_ids = args.client_id_list
    }
    this.group_server_id_list = null
    if (args.group_server_id_list != null)
      this.group_server_id_list = args.group_server_id_list

    if (args.server_id)
      this.server_id = args.server_id

    else
      this.server_id = 0

    if (args.client_id != null)
      this.edge_id = args.client_id

    this.observers = []
    if (client_rank == null)
      this._client_id = `mqttjs_${Math.random().toString(16).substr(2, 8)}`

    else
      this._client_id = String(client_rank)

    this.client_num = client_num
    console.log('mqtt_s3.init: client_num = ', client_num)
    this.set_config_from_objects(config_path)
    this.client_active_list = {}
    this.last_will_msg = JSON.stringify({
      ID: this.edge_id,
      status: CommunicationConstants.MSG_CLIENT_STATUS_OFFLINE,
    })
    this.top_active_msg = CommunicationConstants.CLIENT_TOP_ACTIVE_MSG
    this.topic_last_will_msg = CommunicationConstants.CLIENT_TOP_LAST_WILL_MSG
    this.last_will_msg = JSON.stringify({
      ID: this.edge_id,
      status: CommunicationConstants.MSG_CLIENT_STATUS_OFFLINE,
    })
    this.options = {
      username: this.config_path.MQTT_USER,
      password: this.config_path.MQTT_PWD,
      cleanSession: false,
      keepAlive: 180,
      clientId: `mqttjs_${Math.random().toString(16).substr(2, 8)}`,
      connectTimeout: 15000,
      will: {
        topic: this.topic_last_will_msg,
        payload: this.last_will_msg,
        qos: 2,
        retain: true,
      },
    }
    const connectUrl = mqttUrl
    this.mqtt_mgr = mqtt.connect(connectUrl, this.options)
    // this.mqtt_mgr = currClient.value;
    this.mqtt_mgr.on('connect', () => {
      console.log('Connected')
      this.on_connect()
      this.callback_connected_listener(this.mqtt_mgr)
    })
    // eslint-disable-next-line @typescript-eslint/no-misused-promises
    this.mqtt_mgr.on('message', async (topic, message, packet) => {
      // message is Buffer
      // this.on_message(message);
      console.log('ML Weights Topic: ', topic)
      if (packet.retain) {
        console.log('Received retain message, just return.')
        return
      }
      if (topic == `${this._topic}${String(this.server_id)}_${String(this.client_real_ids[0])}`) {
        const msgData = message.toString()
        const jsonData = JSON.parse(msgData)
        console.log('mqtt-jsonData ', jsonData)
        // this.mqtt_mgr.add_message_passthrough_listener(this.on_message(message));
        const s3_key_str = jsonData[Message.MSG_ARG_KEY_MODEL_PARAMS]
        console.log('message')
        // s3_key_str = s3_key_str.replace(' ', '');
        console.log('receive s3_key: ', s3_key_str)
        if (s3_key_str != undefined) {
          console.log('mqtt_s3.on_message: use s3 pack, s3 message key ', s3_key_str)
          const read_s3_model = await this.s3_storage.read_model(s3_key_str)
          // debugger;
          console.log('read_s3_server_model: ', read_s3_model)
          if (!read_s3_model || !read_s3_model.Body)
            return

          const msgData = read_s3_model.Body.toString()
          const json = JSON.parse(msgData)
          console.log('S3 Response Body was', json)
          const py_layers = Object.keys(json)
          const newWeights: Tensor<Rank>[] = []
          py_layers.forEach((item) => {
            const x = transpose(new Tensor(json[item]))
            newWeights.push(x)
          })
          console.log('get the server model weights: ', newWeights)
          const model_params = newWeights
          console.log(
            'mqtt_s3.on_message: receive model_params: ',
            model_params,
          )
          jsonData[Message.MSG_ARG_KEY_MODEL_PARAMS] = model_params
          this._notify(jsonData)
          console.log(message.toString())
        }
        else {
          console.log('mqtt_s3.on_message: not use s3 pack')
        }
      }
    })
    // this.mqtt_mgr.on_disconnect = self.on_disconnect;

    this.is_connected = false
  }

  // on_message(msg) {
  //   const msgData = msg.toString();
  //   const jsonData = JSON.parse(msgData);
  //   this.mqtt_mgr.add_message_passthrough_listener(this.on_message(msg));
  //   const s3_key_str = jsonData[Message.MSG_ARG_KEY_MODEL_PARAMS];
  //   this._notify(jsonData);
  // }

  callback_published_listener(client: mqtt.MqttClient) {
    this._published_listeners.forEach((listener) => {
      listener(client)
    })
  }

  callback_connected_listener(client: mqtt.MqttClient) {
    this._connected_listeners.forEach((listener) => {
      console.log('listener ', listener)
      listener(client)
    })
  }

  callback_subscribed_listener(client: mqtt.MqttClient) {
    this._subscribed_listeners.forEach((listener) => {
      listener(client)
    })
  }

  _on_subscribe(client: mqtt.MqttClient) {
    this.callback_subscribed_listener(client)
  }

  on_connect() {
    if (this.is_connected)
      return

    const real_topic
      = `${this._topic
      + String(this.server_id)
      }_${
      String(this.client_real_ids[0])}`
    console.log('real_topic ', real_topic)
    this.mqtt_mgr.subscribe(real_topic, { qos: 2 })
    this._notify_connection_ready()

    this.is_connected = true
  }

  on_disconnect() {
    this.is_connected = false
  }

  get_client_id() {
    return this._client_id
  }

  get_topic() {
    return this._topic
  }

  _notify_connection_ready() {
    const msg_params = new Message()
    const msg_type = CommunicationConstants.MSG_TYPE_CONNECTION_IS_READY
    this.observers.forEach((item) => {
      console.log('observer: ', item)
      item.receive_message(msg_type, msg_params)
    })
  }

  _notify(msg_obj: any) {
    const msg_params = new Message()
    msg_params.init_from_json_object(msg_obj)
    const msg_type = msg_params.get_type()
    this.observers.forEach((item) => {
      console.log('receive_message type: ', msg_type)
      item.receive_message(msg_type, msg_params)
    })
  }

  set_config_from_objects(mqtt_config: { BROKER_HOST: any; BROKER_PORT: any; MQTT_USER: null; MQTT_PWD: null }) {
    this.broker_host = mqtt_config.BROKER_HOST
    this.broker_port = mqtt_config.BROKER_PORT
    this.mqtt_user = null
    this.mqtt_pwd = null
    if (mqtt_config.MQTT_USER != null)
      this.mqtt_user = mqtt_config.MQTT_USER

    if (mqtt_config.MQTT_PWD != null)
      this.mqtt_pwd = mqtt_config.MQTT_PWD
  }

  async send_message(msg: Message) {
    console.log('send message ', msg)
    const sender_id = msg.get_sender_id()
    const receiver_id = msg.get_receiver_id()
    console.log('receiver_id ', receiver_id, ' ', 'sender_id ', sender_id)
    const topic = `fedml_${this.topic_in}_${sender_id}`
    console.log('send message topic ', topic)
    const uuid = uuidv4()
    const message_key = `${topic}_${uuid}`
    const payload = msg.get_params()
    const model_params_object = payload[Message.MSG_ARG_KEY_MODEL_PARAMS]
    payload.msg_type = parseInt(String(payload.msg_type))
    payload.deviceType = 'web'
    console.log('topic s3: ', topic)
    // model_params_object = tf_model;
    console.log('model_params_object ', model_params_object)
    if (model_params_object !== undefined) {
      console.log(
        'mqtt_s3.send_message: S3+MQTT msg sent, message_key: ',
        message_key,
      )
      const model_url = await this.s3_storage.write_model(
        message_key,
        model_params_object,
      )
      const model_params_key_url = {
        key: message_key,
        url: model_url,
        obj: model_params_object,
      }
      payload[Message.MSG_ARG_KEY_MODEL_PARAMS] = model_params_key_url.key
      payload[Message.MSG_ARG_KEY_MODEL_PARAMS_URL] = model_params_key_url.url
    }
    setTimeout(() => {
      // const model_params_obj = payload[Message.MSG_ARG_KEY_MODEL_PARAMS];
      this.mqtt_mgr.publish(
        topic,
        JSON.stringify(payload),
        { qos: 2, retain: true },
        (error) => {
          if (error)
            console.log(error)

          else
            console.log('Published')
        },
      )
    }, 6000)
    return payload[Message.MSG_ARG_KEY_MODEL_PARAMS_URL]
  }

  add_observer(observer: Observer) {
    this.observers.push(observer)
  }

  remove_observer(_observer: Observer) {
    throw new Error('Method not implemented.')
  }

  handle_receive_message() {
    console.log('ListenStart')
    // @ts-ignore
    this.mqtt_mgr.handleMessage()
  }

  stop_receive_message() {
    console.log('stop receive message')
  }
}

export default MqttS3MultiClientsCommManager
