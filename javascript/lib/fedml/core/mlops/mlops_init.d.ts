export declare class MLOpsStore {
    mlops_args: Object;
    mlops_project_id: Number;
    mlops_run_id: any;
    mlops_edge_id: any;
    mlops_log_metrics: Object;
    mlops_log_round_info: Object;
    mlops_log_client_training_status: string;
    mlops_log_round_start_time: any;
    mlops_log_metrics_lock: any;
    mlops_log_mqtt_mgr: any;
    mlops_log_mqtt_lock: any;
    mlops_log_mqtt_is_connected: boolean;
    mlops_log_agent_config: Object;
    mlops_metrics: any;
    mlops_event: any;
    mlops_bind_result: boolean;
    server_agent_id: any;
    current_parrot_process: any;
    pre_setup(args: any): void;
    init(args: any): void;
    event(event_name: any, event_started: boolean | undefined, event_value: any, event_edge_id: any): void;
    set_realtime_params(): boolean;
    mlops_enabled(args: any): Boolean;
}
