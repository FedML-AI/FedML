export class Message {
    static MSG_ARG_KEY_OPERATION = "operation"
    static MSG_ARG_KEY_TYPE = "msg_type"
    static MSG_ARG_KEY_SENDER = "sender"
    static MSG_ARG_KEY_RECEIVER = "receiver"

    static MSG_OPERATION_SEND = "send"
    static MSG_OPERATION_RECEIVE = "receive"
    static MSG_OPERATION_BROADCAST = "broadcast"
    static MSG_OPERATION_REDUCE = "reduce"

    static MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    static MSG_ARG_KEY_MODEL_PARAMS_URL = "model_params_url"

    type;
    sender_id;
    receiver_id;
    msg_params;

    constructor(type="default", sender_id=0, receiver_id=0) {
        this.type = String(type);
        this.sender_id = sender_id;
        this.receiver_id = receiver_id;
        this.msg_params[Message.MSG_ARG_KEY_TYPE] = type
        this.msg_params[Message.MSG_ARG_KEY_SENDER] = sender_id
        this.msg_params[Message.MSG_ARG_KEY_RECEIVER] = receiver_id
    }

    init(msg_params) {
        this.msg_params = msg_params;
    }

    init_from_json_string(json_string) {
        this.msg_params = JSON.parse(json_string);
        this.type = this.msg_params[Message.MSG_ARG_KEY_TYPE]
        this.sender_id = this.msg_params[Message.MSG_ARG_KEY_SENDER]
        this.receiver_id = this.msg_params[Message.MSG_ARG_KEY_RECEIVER]
    }

    init_from_json_object(json_object){
        this.msg_params = json_object;
        this.type = this.msg_params[Message.MSG_ARG_KEY_TYPE]
        this.sender_id = this.msg_params[Message.MSG_ARG_KEY_SENDER]
        this.receiver_id = this.msg_params[Message.MSG_ARG_KEY_RECEIVER]
    }

    get_sender_id(){
        return this.sender_id;
    }

    get_receiver_id(){
        return this.receiver_id;
    }

    add_params(key, value){
        this.msg_params[key] = value;
    }

    get_params(){
        return this.msg_params;
    }

    add(key, value) {
        this.msg_params[key] = value;
    }


}