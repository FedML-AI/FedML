export declare class MqttManager {
    _client: any;
    _published_listeners: any[];
    _subscribed_listeners: any[];
    _passthrough_listeners: any[];
    _connected_listeners: any[];
    _listeners: any;
    constructor(connectUrl: any, mqtt_config: any);
    add_message_passthrough_listener(listener: any): void;
    remove_message_passthrough_listener(listener: any): void;
    on_connect(client: any): void;
    on_publish(client: any): void;
    on_disconnect(packet: any): void;
    on_message(msg: any): void;
    on_subscribe(client: any): void;
    callback_connected_listener(client: any): void;
    callback_published_listener(client: any): void;
    callback_subscribed_listener(client: any): void;
    add_connected_listener(listener: any): void;
    remove_connected_listener(listener: any): void;
    add_published_listener(listener: any): void;
    remove_published_listener(listener: any): void;
}
