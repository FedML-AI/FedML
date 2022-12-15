# FedML Octopus - Example with Defense + MNIST + Logistic Regression 

This example illustrates how to add defenses on cross-silo federated learning with FedML Octopus. The source code locates at [https://github.com/FedML-AI/FedML/tree/master/python/examples/cross_silo/mqtt_s3_fedavg_defense_mnist_lr_example](https://github.com/FedML-AI/FedML/tree/master/python/examples/cross_silo/mqtt_s3_fedavg_defense_mnist_lr_example). We use FoolsGold defense for example.


> **If you have multiple nodes, you should run the client script on each node**

## One line API

`python/examples/cross_silo/mqtt_s3_fedavg_defense_mnist_lr_example/one_line`

The highly encapsulated server and client API calls are shown as below:

`run_server.sh` is as follows:

```shell
#!/usr/bin/env bash

python3 server/torch_server.py --cf config/fedml_config.yaml --rank 0
```

`server/torch_server.py`

```python
import fedml


if __name__ == "__main__":
    fedml.run_cross_silo_server()
```

`run_client.sh`


```shell
#!/usr/bin/env bash
RANK=$1
python3 torch_client.py --cf config/fedml_config.yaml --rank $RANK
```

`client/torch_client.py`

```python
import fedml

if __name__ == "__main__":
    fedml.run_cross_silo_client()
```

At the client side, the client ID (a.k.a rank) starts from 1. Please also modify config/fedml_config.yaml, changing the `worker_num` the as the number of clients you plan to run.

At the server side, run the following script:
```
bash run_server.sh
```

For client 1, run the following script:
```
bash run_client.sh 1
```
For client 2, run the following script:
```
bash run_client.sh 2
```
For client 3, run the following script:
```
bash run_client.sh 3
```
For client 4, run the following script:
```
bash run_client.sh 4
```


`config/fedml_config.yaml` is shown below.

Here `common_args.training_type` is "cross_silo" type, and `train_args.client_id_list` needs to correspond to the client id in the client command line.

```yaml
common_args:
  training_type: "cross_silo"
  scenario: "horizontal"
  using_mlops: false
  random_seed: 0
  config_version: release

environment_args:
  bootstrap: config/bootstrap.sh

data_args:
  dataset: "mnist"
  data_cache_dir: ~/fedml_data
  partition_method: "hetero"
  partition_alpha: 0.5

model_args:
  model: "lr"
  model_file_cache_folder: "./model_file_cache" # will be filled by the server automatically
  global_model_file_path: "./model_file_cache/global_model.pt"

train_args:
  federated_optimizer: "FedAvg"
  # for CLI running, this can be None; in MLOps deployment, `client_id_list` will be replaced with real-time selected devices
  client_id_list:
  client_num_in_total: 4
  client_num_per_round: 4
  comm_round: 10
  epochs: 1
  batch_size: 10
  client_optimizer: sgd
  learning_rate: 0.03
  weight_decay: 0.001

validation_args:
  frequency_of_the_test: 1

device_args:
  using_gpu: false
  gpu_mapping_file: config/gpu_mapping.yaml
  gpu_mapping_key: mapping_default

comm_args:
  backend: "MQTT_S3"
  mqtt_config_path:
  s3_config_path:
  grpc_ipconfig_path: ./config/grpc_ipconfig.csv

tracking_args:
   # When running on MLOps platform(open.fedml.ai), the default log path is at ~/fedml-client/fedml/logs/ and ~/fedml-server/fedml/logs/
  enable_wandb: false
  wandb_key: ee0b5f53d949c84cee7decbe7a629e63fb2f8408
  wandb_project: fedml
  wandb_name: fedml_torch_fedavg_mnist_lr

defense_args:
  enable_defense: true
  defense_type: foolsgold
  use_memory: true
```

### Training Results

At the end of training, the server window will see the following output:

```shell
[FedML-Server(0) @device-id-0] [Sat, 27 Aug 2022 23:23:46] [INFO] [fedml_server_manager.py:200:send_message_finish]  ====================send cleanup message to 879====================
[FedML-Server(0) @device-id-0] [Sat, 27 Aug 2022 23:23:46] [INFO] [mqtt_s3_multi_clients_comm_manager.py:230:send_message] mqtt_s3.send_message: msg topic = fedml_123_0_3
[FedML-Server(0) @device-id-0] [Sat, 27 Aug 2022 23:23:46] [INFO] [mqtt_s3_multi_clients_comm_manager.py:255:send_message] mqtt_s3.send_message: MQTT msg sent
[FedML-Server(0) @device-id-0] [Sat, 27 Aug 2022 23:23:47] [INFO] [fedml_server_manager.py:200:send_message_finish]  ====================send cleanup message to 57====================
[FedML-Server(0) @device-id-0] [Sat, 27 Aug 2022 23:23:47] [INFO] [mqtt_s3_multi_clients_comm_manager.py:230:send_message] mqtt_s3.send_message: msg topic = fedml_123_0_4
[FedML-Server(0) @device-id-0] [Sat, 27 Aug 2022 23:23:47] [INFO] [mqtt_s3_multi_clients_comm_manager.py:255:send_message] mqtt_s3.send_message: MQTT msg sent
[FedML-Server(0) @device-id-0] [Sat, 27 Aug 2022 23:23:48] [INFO] [fedml_server_manager.py:200:send_message_finish]  ====================send cleanup message to 496====================
[FedML-Server(0) @device-id-0] [Sat, 27 Aug 2022 23:23:51] [INFO] [fedml_comm_manager.py:62:finish] __finish
[FedML-Server(0) @device-id-0] [Sat, 27 Aug 2022 23:23:51] [INFO] [mqtt_s3_multi_clients_comm_manager.py:296:stop_receive_message] mqtt_s3.stop_receive_message: stopping...
[FedML-Server(0) @device-id-0] [Sat, 27 Aug 2022 23:23:51] [INFO] [fedml_comm_manager.py:28:run] finished...
```


At the end of training, each client window will see the following output:


```shell
[FedML-Client(1) @device-id-1] [Sat, 27 Aug 2022 23:23:51] [INFO] [mqtt_s3_multi_clients_comm_manager.py:191:_on_message_impl] mqtt_s3.on_message: use s3 pack, s3 message key fedml_123_0_1_d7361ad6-a063-48be-b08a-ef0aa3f9ba48
[FedML-Client(1) @device-id-1] [Sat, 27 Aug 2022 23:23:51] [INFO] [mqtt_s3_multi_clients_comm_manager.py:197:_on_message_impl] mqtt_s3.on_message: model params length 2
[FedML-Client(1) @device-id-1] [Sat, 27 Aug 2022 23:23:51] [INFO] [mqtt_s3_multi_clients_comm_manager.py:178:_notify] mqtt_s3.notify: msg type = 2
[FedML-Client(1) @device-id-1] [Sat, 27 Aug 2022 23:23:51] [INFO] [fedml_comm_manager.py:34:receive_message] receive_message. msg_type = 2, sender_id = 0, receiver_id = 1
[FedML-Client(1) @device-id-1] [Sat, 27 Aug 2022 23:23:51] [INFO] [fedml_client_master_manager.py:81:handle_message_receive_model_from_server] handle_message_receive_model_from_server.
[FedML-Client(1) @device-id-1] [Sat, 27 Aug 2022 23:23:51] [INFO] [mqtt_s3_multi_clients_comm_manager.py:204:_on_message_impl] mqtt_s3.on_message: not use s3 pack
[FedML-Client(1) @device-id-1] [Sat, 27 Aug 2022 23:23:51] [INFO] [mqtt_s3_multi_clients_comm_manager.py:178:_notify] mqtt_s3.notify: msg type = 7
[FedML-Client(1) @device-id-1] [Sat, 27 Aug 2022 23:23:51] [INFO] [fedml_comm_manager.py:34:receive_message] receive_message. msg_type = 7, sender_id = 0, receiver_id = 1
[FedML-Client(1) @device-id-1] [Sat, 27 Aug 2022 23:23:51] [INFO] [fedml_client_master_manager.py:98:handle_message_finish]  ====================cleanup ====================
[FedML-Client(1) @device-id-1] [Sat, 27 Aug 2022 23:23:51] [INFO] [fedml_comm_manager.py:62:finish] __finish
[FedML-Client(1) @device-id-1] [Sat, 27 Aug 2022 23:23:51] [INFO] [mqtt_s3_multi_clients_comm_manager.py:296:stop_receive_message] mqtt_s3.stop_receive_message: stopping...
[FedML-Client(1) @device-id-1] [Sat, 27 Aug 2022 23:23:51] [INFO] [fedml_comm_manager.py:28:run] finished...
```

## Five lines of APIs

`torch_client.py`

```python
import fedml
from fedml.cross_silo import Client

if __name__ == "__main__":
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    client = Client(args, device, dataset, model)
    client.run()
```

## Custom data and model 

The custom data and model example locates at the following folder:

`python/examples/cross_silo/mqtt_s3_fedavg_defense_mnist_lr_example/custum_data_and_model`


```python
def load_data(args):
    download_mnist(args.data_cache_dir)
    fedml.logger.info("load_data. dataset_name = %s" % args.dataset)

    """
    Please read through the data loader at to see how to customize the dataset for FedML framework.
    """
    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_partition_data_mnist(
        args.batch_size,
        train_path=args.data_cache_dir + "MNIST/train",
        test_path=args.data_cache_dir + "MNIST/test",
    )
    """
    For shallow NN or linear models, 
    we uniformly sample a fraction of clients each round (as the original FedAvg paper)
    """
    args.client_num_in_total = client_num
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs
```



```python
if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args)

    # load model (the size of MNIST image is 28 x 28)
    model = LogisticRegression(28 * 28, output_dim)

    # start training
    client = Client(args, device, dataset, model)
    client.run()
```


## A Better User-experience with FedML MLOps (open.fedml.ai)
To reduce the difficulty and complexity of these CLI commands. We recommend you to use our MLOps (open.fedml.ai).
FedML MLOps provides:
- Install Client Agent and Login
- Inviting Collaborators and group management
- Project Management
- Experiment Tracking (visualizing training results)
- monitoring device status
- visualizing system performance (including profiling flow chart)
- distributed logging
- model serving

For more details, please refer to [MLOps User Guide](./../../mlops/user_guide.md).