declare const runArgsJSON: {
    configName: string;
    packages_config: {
        linuxClient: string;
        server: string;
        androidClientVersion: number;
        androidClientUrl: string;
        serverUrl: string;
        androidClient: string;
        linuxClientUrl: string;
    };
    data_config: {
        privateLocalData: string;
        syntheticDataUrl: string;
        syntheticData: string;
    };
    userId: string;
    parameters: {
        model_args: {
            model_file_cache_folder: string;
            model: string;
            global_model_file_path: string;
        };
        device_args: {
            using_gpu: boolean;
            gpu_mapping_key: string;
            gpu_mapping_file: string;
        };
        comm_args: {
            grpc_ipconfig_path: string;
            backend: string;
        };
        train_args: {
            batch_size: number;
            weight_decay: number;
            client_num_per_round: number;
            client_num_in_total: number;
            comm_round: number;
            client_optimizer: string;
            epochs: number;
            learning_rate: number;
            federated_optimizer: string;
        };
        environment_args: {
            bootstrap: string;
        };
        validation_args: {
            frequency_of_the_test: number;
        };
        common_args: {
            random_seed: number;
            scenario: string;
            training_type: string;
            config_version: string;
            using_mlops: boolean;
        };
        data_args: {
            partition_method: string;
            partition_alpha: number;
            dataset: string;
            data_cache_dir: string;
        };
        tracking_args: {
            wandb_project: string;
            wandb_name: string;
            wandb_key: string;
            enable_wandb: boolean;
            log_file_dir: string;
        };
    };
};
export type RunArgsJSON = Partial<typeof runArgsJSON>;
export declare class Arguments<T = any> {
    worker_num: number | undefined;
    client_num_per_round: number;
    data_cache_dir: any;
    data_file_path: any;
    partition_file_path: any;
    part_file: any;
    rank: string | number;
    constructor(cmd_args: T, training_type?: null, comm_backend?: null, override_cmd_args?: boolean);
    load_yaml_config(): Promise<any>;
    get_default_yaml_config(cmd_args: T, training_type?: null, comm_backend?: null): Promise<any>;
    set_attr_from_config(configuration: {
        [x: string]: {
            [x: string]: any;
        };
    }): void;
    static add_args(): {
        yaml_config_file: string;
        run_id: number;
        rank: number;
        local_rank: number;
        node_rank: number;
        role: string;
    };
    static load_arguments(training_type?: null, comm_backend?: null): Arguments<{
        yaml_config_file: string;
        run_id: number;
        rank: number;
        local_rank: number;
        node_rank: number;
        role: string;
    }>;
}
export {};
