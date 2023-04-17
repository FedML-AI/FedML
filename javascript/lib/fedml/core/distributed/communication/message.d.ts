export declare class Message {
    static MSG_ARG_KEY_OPERATION: string;
    static MSG_ARG_KEY_TYPE: string;
    static MSG_ARG_KEY_SENDER: string;
    static MSG_ARG_KEY_RECEIVER: string;
    static MSG_OPERATION_SEND: string;
    static MSG_OPERATION_RECEIVE: string;
    static MSG_OPERATION_BROADCAST: string;
    static MSG_OPERATION_REDUCE: string;
    static MSG_ARG_KEY_MODEL_PARAMS: string;
    static MSG_ARG_KEY_MODEL_PARAMS_URL: string;
    type: string;
    sender_id: number;
    receiver_id: number;
    msg_params: {
        msg_type: string | number;
        sender: number;
        receiver: number;
        deviceType: string;
    };
    constructor(type?: number, sender_id?: number, receiver_id?: number, device_type?: string);
    init(msg_params: any): void;
    init_from_json_string(json_string: any): void;
    init_from_json_object(json_object: any): void;
    get_sender_id(): number;
    get_receiver_id(): number;
    add_params(key: any, value: any): void;
    get_params(): {
        msg_type: string | number;
        sender: number;
        receiver: number;
        deviceType: string;
    };
    add(key: any, value: any): void;
    get(key: any): any;
    get_type(): string | number;
    to_string(): {
        msg_type: string | number;
        sender: number;
        receiver: number;
        deviceType: string;
    };
    to_json(): string;
    get_content(): {
        msg_type: string | number;
        sender: number;
        receiver: number;
        deviceType: string;
    };
}
