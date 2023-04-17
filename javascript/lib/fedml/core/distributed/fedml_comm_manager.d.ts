import { BaseCommunicationManager } from './communication/base_com_manager';
export declare class FedMLCommManager {
    args: any;
    comm: any;
    rank: number;
    size: number;
    backend: string;
    com_manager: any;
    message_handler_dict: {};
    constructor(args: any, comm?: null, rank?: number, size?: number, backend?: string);
    register_comm_manager(comm_manager: BaseCommunicationManager): void;
    run(): void;
    get_sender_id(): number;
    send_message(message: any): Promise<any>;
    register_message_receive_handler(msg_type: any, handler_callback_func: any): void;
    finish(): void;
    init_manager(): Promise<void>;
    get_training_mqtt_s3_config(): Promise<{
        mqtt_config: any;
        s3_config: any;
    }>;
    receive_message(msg_type: any, msg_params: any): void;
    send_client_status(receive_id: any, status?: string): Promise<void>;
}
export default FedMLCommManager;
