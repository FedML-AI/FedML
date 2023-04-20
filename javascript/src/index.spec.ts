// import { describe, expect, it } from 'vitest'
import { fedml_train } from './index'

const run_args = {
  cloudServerDeviceId: 'ef0a108689a8443a9de1ce61f301519e@Linux.Public.Cloud.ServerInstance',
  threshold: '20',
  starttime: 1681113496267,
  edgestates: '{}',
  edgeids: [
    188527,
  ],
  urls: '"[]"',
  id: 3637,
  state: 'STARTING',
  projectid: 268,
  run_config: {
    configName: 'webjs-mnist',
    packages_config: {
      linuxClient: 'client-package.zip',
      server: 'server-package.zip',
      androidClientVersion: 0,
      androidClientUrl: '',
      serverUrl: 'https://fedml.s3.us-west-1.amazonaws.com/1667442853400server-package.zip',
      androidClient: '',
      linuxClientUrl: 'https://fedml.s3.us-west-1.amazonaws.com/1667442859083client-package.zip',
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
        is_browser: true,
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
        comm_round: 2,
        client_optimizer: 'sgd',
        epochs: 1,
        learning_rate: 0.03,
        federated_optimizer: 'FedAvg',
      },
      environment_args: {
        bootstrap: 'config/bootstrap.sh',
      },
      validation_args: {
        frequency_of_the_test: 1,
      },
      common_args: {
        random_seed: 0,
        scenario: 'horizontal',
        training_type: 'cross_silo',
        config_version: 'release',
        using_mlops: true,
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
  },
  timestamp: '1681113496334',
  cloud_agent_id: '226',
  lastupdatetime: 1681113496291,
  create_time: 1681113496291,
  groupid: 171,
  edges: [
    {
      device_id: '538247af806608e9f868adf6cd53d509@Browser.Edge.Device',
      os_type: 'Browser',
      id: 188527,
    },
  ],
  server_id: 188528,
  token: 'eyJhbGciOiJIUzI1NiJ9.eyJpZCI6MjA5LCJhY2NvdW50IjoiZmVkbWwteWlkZSIsImxvZ2luVGltZSI6IjE2ODExMTM0MTExMjgiLCJleHAiOjE2ODE3MTgyMTF9.rOh0x_OQhO7xv2th82vLLymu_9dRJkot_2V3BGDWebA',
  name: 'determine_speed',
  creater: '209',
  runId: 3637,
  applicationId: 185,
  group_server_id_list: [
    188528,
  ],
  status: 0,
  is_retain: 0,
  currentEdgeId: 188527,
}

await fedml_train(run_args, 0)

// describe('fedml_train', () => {
//   it('fedml_train', async () => {
//     console.log(result)
//     expect(result).toBe(undefined)
//   })
// })
