export class Message {
  static MSG_ARG_KEY_OPERATION = 'operation';
  static MSG_ARG_KEY_TYPE = 'msg_type';
  static MSG_ARG_KEY_SENDER = 'sender';
  static MSG_ARG_KEY_RECEIVER = 'receiver';

  static MSG_OPERATION_SEND = 'send';
  static MSG_OPERATION_RECEIVE = 'receive';
  static MSG_OPERATION_BROADCAST = 'broadcast';
  static MSG_OPERATION_REDUCE = 'reduce';

  static MSG_ARG_KEY_MODEL_PARAMS = 'model_params';
  static MSG_ARG_KEY_MODEL_PARAMS_URL = 'model_params_url';

  type: string | number;
  sender_id: number;
  receiver_id: number;
  msg_params: {
    msg_type: string | number;
    sender: number;
    receiver: number;
    deviceType: string;
    [s: string]: any;
  };

  constructor(type = 0, sender_id = 0, receiver_id = 0, device_type = 'web') {
    this.type = String(type);
    this.sender_id = sender_id;
    this.receiver_id = receiver_id;
    this.msg_params = {
      msg_type: String(type),
      sender: sender_id,
      receiver: receiver_id,
      deviceType: device_type,
    };
  }

  init(msg_params: {
    msg_type: string | number;
    sender: number;
    receiver: number;
    deviceType: string;
  }) {
    this.msg_params = msg_params;
  }

  init_from_json_string(json_string: string) {
    this.msg_params = JSON.parse(json_string);
    this.type = this.msg_params.msg_type;
    this.sender_id = this.msg_params.sender;
    this.receiver_id = this.msg_params.receiver;
  }

  init_from_json_object(json_object: {
    msg_type: string | number;
    sender: number;
    receiver: number;
    deviceType: string;
  }) {
    this.msg_params = json_object;
    this.type = this.msg_params.msg_type;
    this.sender_id = this.msg_params.sender;
    this.receiver_id = this.msg_params.receiver;
  }

  get_sender_id() {
    return this.sender_id;
  }

  get_receiver_id() {
    return this.receiver_id;
  }

  add_params(key: string | number, value: string | number) {
    this.msg_params[key] = value;
  }

  get_params() {
    return this.msg_params;
  }

  add(key: keyof typeof this.msg_params, value: any) {
    this.msg_params[key] = value;
  }

  get(key: keyof typeof this.msg_params) {
    if (this.msg_params != undefined || this.msg_params[key] != null) return this.msg_params[key];

    return null;
  }

  get_type() {
    return this.msg_params.msg_type;
  }

  to_string() {
    return this.msg_params;
  }

  to_json() {
    const json_string = JSON.stringify(this.msg_params);
    return json_string;
  }

  get_content() {
    return this.msg_params;
  }
}
