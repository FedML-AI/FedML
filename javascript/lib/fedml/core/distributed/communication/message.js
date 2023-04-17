class Message {
  constructor(type = 0, sender_id = 0, receiver_id = 0, device_type = 'web') {
    this.type = String(type)
    this.sender_id = sender_id
    this.receiver_id = receiver_id
    this.msg_params = {
      msg_type: String(type),
      sender: sender_id,
      receiver: receiver_id,
      deviceType: device_type,
    }
  }

  init(msg_params) {
    this.msg_params = msg_params
  }

  init_from_json_string(json_string) {
    this.msg_params = JSON.parse(json_string)
    this.type = this.msg_params.msg_type
    this.sender_id = this.msg_params.sender
    this.receiver_id = this.msg_params.receiver
  }

  init_from_json_object(json_object) {
    this.msg_params = json_object
    this.type = this.msg_params.msg_type
    this.sender_id = this.msg_params.sender
    this.receiver_id = this.msg_params.receiver
  }

  get_sender_id() {
    return this.sender_id
  }

  get_receiver_id() {
    return this.receiver_id
  }

  add_params(key, value) {
    this.msg_params[key] = value
  }

  get_params() {
    return this.msg_params
  }

  add(key, value) {
    this.msg_params.key = value
  }

  get(key) {
    if (this.msg_params != undefined || this.msg_params.key != null)
      return this.msg_params.key
    return null
  }

  get_type() {
    return this.msg_params.msg_type
  }

  to_string() {
    return this.msg_params
  }

  to_json() {
    const json_string = JSON.stringify(this.msg_params)
    return json_string
  }

  get_content() {
    return this.msg_params
  }
}
Message.MSG_ARG_KEY_OPERATION = 'operation'
Message.MSG_ARG_KEY_TYPE = 'msg_type'
Message.MSG_ARG_KEY_SENDER = 'sender'
Message.MSG_ARG_KEY_RECEIVER = 'receiver'
Message.MSG_OPERATION_SEND = 'send'
Message.MSG_OPERATION_RECEIVE = 'receive'
Message.MSG_OPERATION_BROADCAST = 'broadcast'
Message.MSG_OPERATION_REDUCE = 'reduce'
Message.MSG_ARG_KEY_MODEL_PARAMS = 'model_params'
Message.MSG_ARG_KEY_MODEL_PARAMS_URL = 'model_params_url'
export { Message }
