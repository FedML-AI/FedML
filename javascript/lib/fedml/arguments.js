// import { homedir } from 'os';
// import fs from 'fs';
const runArgsJSON = {
  configName: 'newPackage',
  packages_config: {
    linuxClient: 'client-package.zip',
    server: 'server-package.zip',
    androidClientVersion: 0,
    androidClientUrl: '',
    serverUrl: 'https://fedml.s3.us-west-1.amazonaws.com/1658063399409server-package.zip',
    androidClient: '',
    linuxClientUrl: 'https://fedml.s3.us-west-1.amazonaws.com/1658063403141client-package.zip',
  },
  data_config: {
    privateLocalData: '',
    syntheticDataUrl: '',
    syntheticData: '',
  },
  userId: 'fedml-yide',
  parameters: {
    model_args: {
      model_file_cache_folder: './model_file_cache',
      model: 'lr',
      global_model_file_path: './model_file_cache/global_model.pt',
    },
    device_args: {
      using_gpu: false,
      gpu_mapping_key: 'mapping_default',
      gpu_mapping_file: 'config/gpu_mapping.yaml',
    },
    comm_args: {
      grpc_ipconfig_path: './config/grpc_ipconfig.csv',
      backend: 'MQTT_S3',
    },
    train_args: {
      batch_size: 10,
      weight_decay: 0.001,
      client_num_per_round: 2,
      client_num_in_total: 2,
      comm_round: 50,
      client_optimizer: 'sgd',
      epochs: 1,
      learning_rate: 0.03,
      federated_optimizer: 'FedAvg',
    },
    environment_args: { bootstrap: 'config/bootstrap.sh' },
    validation_args: { frequency_of_the_test: 1 },
    common_args: {
      random_seed: 0,
      scenario: 'horizontal',
      training_type: 'cross_silo',
      config_version: 'release',
      using_mlops: false,
    },
    data_args: {
      partition_method: 'hetero',
      partition_alpha: 0.5,
      dataset: 'mnist',
      data_cache_dir: '~/fedml_data',
    },
    tracking_args: {
      wandb_project: 'fedml',
      wandb_name: 'fedml_torch_fedavg_mnist_lr',
      wandb_key: 'ee0b5f53d949c84cee7decbe7a629e63fb2f8408',
      enable_wandb: false,
      log_file_dir: './log',
    },
  },
}
export class Arguments {
  constructor(cmd_args, training_type = null, comm_backend = null, override_cmd_args = true) {
    // Object.keys(cmd_args).map((key) => {
    //   this[key] = cmd_args.key;
    // });
    for (const i in cmd_args)
      this[i] = cmd_args[i]
    this.get_default_yaml_config(cmd_args, training_type, comm_backend)
    if (!override_cmd_args) {
      for (const j in cmd_args)
        this[j] = cmd_args[j]
    }
  }

  async load_yaml_config() {
    try {
      const yaml_config = '{"configName":"newPackage","packages_config":{"linuxClient":"client-package.zip","server":"server-package.zip","androidClientVersion":0,"androidClientUrl":"","serverUrl":"https://fedml.s3.us-west-1.amazonaws.com/1658063399409server-package.zip","androidClient":"","linuxClientUrl":"https://fedml.s3.us-west-1.amazonaws.com/1658063403141client-package.zip"},"data_config":{"privateLocalData":"","syntheticDataUrl":"","syntheticData":""},"userId":"fedml-yide","parameters":{"model_args":{"model_file_cache_folder":"./model_file_cache","model":"lr","global_model_file_path":"./model_file_cache/global_model.pt"},"device_args":{"using_gpu":false,"gpu_mapping_key":"mapping_default","gpu_mapping_file":"config/gpu_mapping.yaml"},"comm_args":{"grpc_ipconfig_path":"./config/grpc_ipconfig.csv","backend":"MQTT_S3"},"train_args":{"batch_size":10,"weight_decay":0.001,"client_num_per_round":2,"client_num_in_total":2,"comm_round":50,"client_optimizer":"sgd","epochs":1,"learning_rate":0.03,"federated_optimizer":"FedAvg"},"environment_args":{"bootstrap":"config/bootstrap.sh"},"validation_args":{"frequency_of_the_test":1},"common_args":{"random_seed":0,"scenario":"horizontal","training_type":"cross_silo","config_version":"release","using_mlops":false},"data_args":{"partition_method":"hetero","partition_alpha":0.5,"dataset":"mnist","data_cache_dir":"~/fedml_data"},"tracking_args":{"wandb_project":"fedml","wandb_name":"fedml_torch_fedavg_mnist_lr","wandb_key":"ee0b5f53d949c84cee7decbe7a629e63fb2f8408","enable_wandb":false,"log_file_dir":"./log"}}}'
      const yamlObj = JSON.parse(yaml_config)
      return yamlObj
    }
    catch (error) {
      console.log(error)
    }
  }

  async get_default_yaml_config(cmd_args, training_type = null, comm_backend = null) {
    if (cmd_args.yaml_config_file == '') {
      if (training_type == 'cross_silo')
        console.log('training_type == FEDML_TRAINING_PLATFORM_CROSS_SILO')
    }
    const configuration = await this.load_yaml_config()
    this.set_attr_from_config(configuration)
    return configuration
  }

  set_attr_from_config(configuration) {
    for (const i in configuration) {
      if (typeof configuration[i] == 'object') {
        for (const j in configuration[i])
          this[j] = configuration[i][j]
      }
    }
  }

  static add_args() {
    const args = {
      yaml_config_file: 'config/fedml_config.yaml',
      run_id: 0,
      rank: 0,
      local_rank: 0,
      node_rank: 0,
      role: 'client',
    }
    return args
  }

  static load_arguments(training_type = null, comm_backend = null) {
    const cmd_args = this.add_args()
    const args = new Arguments(cmd_args, training_type, comm_backend)
    if (args.worker_num == undefined) {
      args.worker_num = 1
      args.client_num_per_round = 1
    }
    // if (args.data_cache_dir != undefined) {
    //   args.data_cache_dir = homedir.path.expanduser(args.data_cache_dir);
    // }
    // if (args.data_file_path != undefined) {
    //   args.data_file_path = homedir.path.expanduser(args.data_file_path);
    // }
    // if (args.partition_file_path != undefined) {
    //   args.partition_file_path = homedir.path.expanduser(args.partition_file_path);
    // }
    // if (args.part_file != undefined) {
    //   args.part_file = homedir.path.expanduser(args.part_file);
    // }
    args.rank = parseInt(args.rank)
    return args
  }
}
