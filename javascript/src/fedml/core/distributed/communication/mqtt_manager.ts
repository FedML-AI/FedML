import mqtt from 'mqtt/dist/mqtt.min'

export class MqttManager {
  _client
  _listeners: Record<string, Function>
  _connected_listeners: Array<Function>
  _subscribed_listeners: Array<Function>
  _published_listeners: Array<Function>
  _passthrough_listeners: Array<Function>

  constructor(connectUrl: string, mqtt_config?: mqtt.IClientOptions) {
    this._published_listeners = []
    this._subscribed_listeners = []
    this._passthrough_listeners = []
    this._connected_listeners = []
    this._listeners = {}
    this._client = mqtt.connect(connectUrl, mqtt_config)
    this._client.on('connect', () => this.on_connect(this._client))
    this._client.subscribe('testtopic', () => this.on_subscribe(this._client))
    this._client.publish('testtopic', 'Hello, MQTT!', () => this.on_publish(this._client))
    this._client.on('disconnect', this.on_disconnect)
    this._client.on('message', (topic, message, packet) => {
      if (packet.retain) {
        console.log('Received retain message, just return.')
        return
      }
      console.log('topic + message', message)
      this._passthrough_listeners.forEach((passthrough_listener) => {
        passthrough_listener(message)
      })
      const _listener = this._listeners[topic]
      if (_listener && typeof _listener === 'function')
        _listener(topic, JSON.stringify(packet.payload))
    })
  }

  private _add_item<T>(list: T[], item?: T) {
    if (item && !list.includes(item))
      list.push(item)
    return this
  }

  private _remove_item<T>(list: T[], item?: T) {
    if (item)
      list = list.filter(l => l !== item)
    return this
  }

  on_connect(client: mqtt.MqttClient) {
    this.callback_connected_listener(client)
  }

  on_publish(client: mqtt.MqttClient) {
    this.callback_published_listener(client)
  }

  on_disconnect(packet: any) {
    console.log(packet)
  }

  on_message(msg: any) {
    this._passthrough_listeners.forEach((passthrough_listener) => {
      passthrough_listener(msg)
    })
  }

  on_subscribe(client: mqtt.MqttClient) {
    this.callback_subscribed_listener(client)
  }

  callback_connected_listener(client: mqtt.MqttClient) {
    this._connected_listeners.forEach((listener) => {
      listener(client)
    })
  }

  callback_published_listener(client: mqtt.MqttClient) {
    this._published_listeners.forEach((listener) => {
      listener(client)
    })
  }

  callback_subscribed_listener(client: mqtt.MqttClient) {
    this._subscribed_listeners.forEach((listener) => {
      listener(client)
    })
  }

  add_connected_listener(listener: Function) {
    this._connected_listeners.push(listener)
  }

  remove_connected_listener(listener?: Function) {
    return this._remove_item(this._connected_listeners, listener)
  }

  add_published_listener(listener: Function) {
    return this._add_item(this._published_listeners, listener)
  }

  remove_published_listener(listener?: Function) {
    return this._remove_item(this._published_listeners, listener)
  }

  add_message_passthrough_listener(listener: Function) {
    this.remove_message_passthrough_listener(listener)
      ._add_item(this._passthrough_listeners, listener)
  }

  remove_message_passthrough_listener(listener: Function) {
    return this._remove_item(this._passthrough_listeners, listener)
  }
}
