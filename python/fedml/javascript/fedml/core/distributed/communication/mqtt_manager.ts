import * as mqtt from 'mqtt/dist/mqtt.min';
export class MqttManager {
  _client: any;
  _published_listeners;
  _subscribed_listeners;
  _passthrough_listeners;
  _connected_listeners;
  _listeners;
  constructor(connectUrl, mqtt_config) {
    this._published_listeners = [];
    this._subscribed_listeners = [];
    this._passthrough_listeners = [];
    this._connected_listeners = [];
    this._client = mqtt.connect(connectUrl, mqtt_config);
    this._client.on('connect', this.on_connect(this._client));
    this._client.subscribe('testtopic', this.on_subscribe(this._client));
    this._client.publish('testtopic', 'Hello, MQTT!', this.on_publish(this._client));
    this._client.on('disconnect', this.on_disconnect);
    this._client.on('message', (topic, message) => {
      console.log('topic + message', message);
      this._passthrough_listeners.forEach((passthrough_listener) => {
        passthrough_listener(message);
      });
      const _listener = this._listeners[message.topic];
      _listener(message.topic, JSON.stringify(message.payload));
    });
  }

  add_message_passthrough_listener(listener) {
    this.remove_message_passthrough_listener(listener);
    this._passthrough_listeners.push(listener);
  }

  remove_message_passthrough_listener(listener) {
    this._passthrough_listeners.pop(listener);
  }

  on_connect(client) {
    this.callback_connected_listener(client);
  }

  on_publish(client) {
    this.callback_published_listener(client);
  }
  on_disconnect(packet) {
    console.log(packet);
  }
  on_message(msg) {
    this._passthrough_listeners.forEach((passthrough_listener) => {
      passthrough_listener(msg);
    });
  }

  on_subscribe(client) {
    this.callback_subscribed_listener(client);
  }

  callback_connected_listener(client) {
    this._connected_listeners.forEach((listener) => {
      listener(client);
    });
  }

  callback_published_listener(client) {
    this._published_listeners.forEach((listener) => {
      listener(client);
    });
  }

  callback_subscribed_listener(client) {
    this._subscribed_listeners.forEach((listener) => {
      listener(client);
    });
  }

  add_connected_listener(listener) {
    this._connected_listeners.push(listener);
  }

  remove_connected_listener(listener) {
    this._connected_listeners.pop(listener);
  }

  add_published_listener(listener) {
    this._published_listeners.push(listener);
  }

  remove_published_listener(listener) {
    this._published_listeners.pop(listener);
  }
}
