# FedML Octopus Example with MNIST + Logistic Regression

This example illustrates how to do real-world hierarchical cross-silo federated learning with FedML Octopus. Hierarchical architecture allows each silo to do the training on multiple trainers in parrallel uisng pytorch's Distributed Data Parallel (DDP). Using this feature your silo can have multiple nodes and run distributed trainers on each node. The source code locates at `python/examples/cross_silo/mqtt_s3_fedavg_hierarchical_mnist_lr_example`.

> :info: **If you have multiple nodes you should run the client script on each node**
 
## One line API

`python/examples/cross_silo/mqtt_s3_fedavg_hierarchical_mnist_lr_example/one_line`

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
    fedml.run_hierarchical_cross_silo_server()
```

`run_client.sh`


```shell
#!/usr/bin/env bash
RANK=$1
python3 client_dist_launcher.py --cf config/fedml_config.yaml --rank $RANK
```

`client/torch_client.py`

```python
import fedml

if __name__ == "__main__":
    fedml.run_hierarchical_cross_silo_client()
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
Note: please run the server first.



`config/fedml_config.yaml` is shown below.

Here `common_args.training_type` is "cross_silo" type, and `train_args.client_id_list` needs to correspond to the client id in the client command line.
You can define your silo's DDP setting using `client_silo_args` where:

  - `n_node_in_silo` is number of nodes in silo
  - `n_proc_per_node` is number of processes (distributed trainers) in each node
  - `node_rank_in_silo`: rank of the node executing the script
  - `pg_master_address` is ip address of the process group master. For all nodes, this should be the ip of the node with rank 0
  - `pg_master_port` is port of the process group master. For all nodes, this should be the port of the node with rank 0

```yaml
common_args:
  training_type: "cross_silo"
  random_seed: 0

data_args:
  dataset: "mnist"
  data_cache_dir: "./../../../data"
  partition_method: "hetero"
  partition_alpha: 0.5

model_args:
  model: "lr"
  model_file_cache_folder: "./model_file_cache" # will be filled by the server automatically
  global_model_file_path: "./model_file_cache/global_model.pt"

train_args:
  federated_optimizer: "FedAvg"
  client_id_list: "[1, 2]"
  client_num_in_total: 1000
  client_num_per_round: 2
  comm_round: 50
  epochs: 1
  batch_size: 10
  client_optimizer: sgd
  learning_rate: 0.03
  weight_decay: 0.001

validation_args:
  frequency_of_the_test: 5

device_args:
  worker_num: 2
  using_gpu: false
  gpu_mapping_file: config/gpu_mapping.yaml
  gpu_mapping_key: mapping_default

comm_args:
  backend: "MQTT_S3"
  mqtt_config_path: config/mqtt_config.yaml
  s3_config_path: config/s3_config.yaml

tracking_args:
  log_file_dir: ./log
  enable_wandb: false
  wandb_key: ee0b5f53d949c84cee7decbe7a629e63fb2f8408
  wandb_project: fedml
  wandb_name: fedml_torch_fedavg_mnist_lr

client_silo_args:
  n_node_in_silo: 1
  n_proc_per_node: 2
  node_rank_in_silo: 0
  pg_master_address: '127.0.0.1'
  pg_master_port: 12345
```

### Training Results

At the end of the 50th training round, the server window will see the following output:

```shell
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 23:54:35] [INFO] [fedml_aggregator.py:197:test_on_server_for_all_clients] ################test_on_server_for_all_clients : 49
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 23:54:35] [INFO] [fedml_server_manager.py:150:handle_message_receive_model_from_client] aggregator_dist_adapter.aggregator.test exception: 'tuple' object has no attribute 'to'
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 23:54:35] [INFO] [fedml_aggregator.py:116:data_silo_selection] client_num_in_total = 1000, client_num_per_round = 2
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 23:54:35] [INFO] [fedml_server_manager.py:214:send_message_sync_model_to_client] send_message_sync_model_to_client. receive_id = 1
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 23:54:35] [INFO] [mqtt_s3_multi_clients_comm_manager.py:240:send_message] mqtt_s3.send_message: starting...
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 23:54:35] [INFO] [mqtt_s3_multi_clients_comm_manager.py:246:send_message] mqtt_s3.send_message: msg topic = fedml_0_0_1
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 23:54:35] [INFO] [mqtt_s3_multi_clients_comm_manager.py:253:send_message] mqtt_s3.send_message: S3+MQTT msg sent, s3 message key = fedml_0_0_1_fb92d0a2-c8cf-4644-b7f6-8ec613987ba5
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 23:54:35] [INFO] [mqtt_s3_multi_clients_comm_manager.py:265:send_message] mqtt_s3.send_message: to python client.
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 23:54:35] [INFO] [fedml_server_manager.py:214:send_message_sync_model_to_client] send_message_sync_model_to_client. receive_id = 2
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 23:54:35] [INFO] [mqtt_s3_multi_clients_comm_manager.py:240:send_message] mqtt_s3.send_message: starting...
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 23:54:35] [INFO] [mqtt_s3_multi_clients_comm_manager.py:246:send_message] mqtt_s3.send_message: msg topic = fedml_0_0_2
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 23:54:35] [INFO] [mqtt_s3_multi_clients_comm_manager.py:253:send_message] mqtt_s3.send_message: S3+MQTT msg sent, s3 message key = fedml_0_0_2_d75fa518-c379-4251-9780-7f229daf7b67
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 23:54:35] [INFO] [mqtt_s3_multi_clients_comm_manager.py:265:send_message] mqtt_s3.send_message: to python client.
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 23:54:36] [INFO] [server_manager.py:121:finish] __finish server
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 23:54:36] [INFO] [mqtt_s3_multi_clients_comm_manager.py:337:stop_receive_message] mqtt_s3.stop_receive_message: stopping...
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 23:54:36] [INFO] [mqtt_s3_multi_clients_comm_manager.py:157:_on_disconnect] mqtt_s3.on_disconnect: disconnection returned result 0, user data None
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 23:54:36] [INFO] [server_manager.py:95:run] running
[FedML-Server(0) @device-id-0] [Sun, 01 May 2022 23:54:36] [INFO] [mqtt_s3_status_manager.py:78:_on_disconnect] mqtt_s3.on_disconnect: disconnection returned result 0, user data None
...
```

At the end of the 50th training round, the client1 window will see the following output:

```shell
edML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [client_master_manager.py:142:handle_message_receive_model_from_server] #######training########### round_id = 49
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [client_slave_manager.py:43:await_sync_process_group] prcoess 1 received round_number 48
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [my_model_trainer_classification.py:45:train] Update Epoch: 0 [10/20 (50%)]   Loss: 2.328646
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [my_model_trainer_classification.py:45:train] Update Epoch: 0 [10/20 (50%)]   Loss: 2.292774
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [my_model_trainer_classification.py:45:train] Update Epoch: 0 [20/20 (100%)]  Loss: 2.282324
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [my_model_trainer_classification.py:45:train] Update Epoch: 0 [20/20 (100%)]  Loss: 2.340256
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [my_model_trainer_classification.py:56:train] Client Index = 0        Epoch: 0        Loss: 2.305485
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [my_model_trainer_classification.py:56:train] Client Index = 0        Epoch: 0        Loss: 2.316515
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [client_manager.py:115:send_message] Sending message (type 3) to server
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [client_slave_manager.py:38:await_sync_process_group] prcoess 1 waiting for round number
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [mqtt_s3_multi_clients_comm_manager.py:240:send_message] mqtt_s3.send_message: starting...
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [mqtt_s3_multi_clients_comm_manager.py:294:send_message] mqtt_s3.send_message: S3+MQTT msg sent, message_key = fedml_0_1_ff61f917-ea9c-4bcc-b233-64b6b01f5e33
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [mqtt_s3_multi_clients_comm_manager.py:306:send_message] mqtt_s3.send_message: to python client.
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [mqtt_s3_multi_clients_comm_manager.py:187:_on_message_impl] --------------------------
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [mqtt_s3_multi_clients_comm_manager.py:200:_on_message_impl] mqtt_s3.on_message: use s3 pack, s3 message key fedml_0_0_1_fb92d0a2-c8cf-4644-b7f6-8ec613987ba5
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [mqtt_s3_multi_clients_comm_manager.py:208:_on_message_impl] mqtt_s3.on_message: from python client.
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [mqtt_s3_multi_clients_comm_manager.py:211:_on_message_impl] mqtt_s3.on_message: model params length 2
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [mqtt_s3_multi_clients_comm_manager.py:182:_notify] mqtt_s3.notify: msg type = 2
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [client_master_manager.py:119:handle_message_receive_model_from_server] handle_message_receive_model_from_server.
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [client_master_manager.py:228:sync_process_group] sending round number to pg
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [client_master_manager.py:233:sync_process_group] round number 49 broadcasted to process group
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [client_master_manager.py:151:finish] Training finished for master client rank 0 in silo 0
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [trainer_dist_adapter.py:133:cleanup_pg] Cleaningup process group for client 0 in silo 0
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [client_manager.py:129:finish] __finish client
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [mqtt_s3_multi_clients_comm_manager.py:337:stop_receive_message] mqtt_s3.stop_receive_message: stopping...
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [mqtt_s3_multi_clients_comm_manager.py:157:_on_disconnect] mqtt_s3.on_disconnect: disconnection returned result 0, user data None
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [client_slave_manager.py:43:await_sync_process_group] prcoess 1 received round_number 49
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [client_master_manager.py:241:run] Connection is ready!
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [trainer_dist_adapter.py:133:cleanup_pg] Cleaningup process group for client 1 in silo 1
[FedML-Client(1) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [client_slave_manager.py:32:finish] Training finsihded for slave client rank 1 in silo 1
...
```

At the end of the 50th training round, the client2 window will see the following output:


```shell
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [client_master_manager.py:142:handle_message_receive_model_from_server] #######training########### round_id = 49
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [client_slave_manager.py:43:await_sync_process_group] prcoess 1 received round_number 48
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [my_model_trainer_classification.py:45:train] Update Epoch: 0 [10/20 (50%)]   Loss: 2.292774
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [my_model_trainer_classification.py:45:train] Update Epoch: 0 [10/20 (50%)]   Loss: 2.328646
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [my_model_trainer_classification.py:45:train] Update Epoch: 0 [20/20 (100%)]  Loss: 2.340256
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [my_model_trainer_classification.py:56:train] Client Index = 1        Epoch: 0        Loss: 2.316515
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [my_model_trainer_classification.py:45:train] Update Epoch: 0 [20/20 (100%)]  Loss: 2.282324
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [client_slave_manager.py:38:await_sync_process_group] prcoess 1 waiting for round number
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [my_model_trainer_classification.py:56:train] Client Index = 1        Epoch: 0        Loss: 2.305485
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [client_manager.py:115:send_message] Sending message (type 3) to server
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [mqtt_s3_multi_clients_comm_manager.py:240:send_message] mqtt_s3.send_message: starting...
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [mqtt_s3_multi_clients_comm_manager.py:294:send_message] mqtt_s3.send_message: S3+MQTT msg sent, message_key = fedml_0_2_a809c1e6-0461-47fa-a5ce-7693ffb9e5ab
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:35] [INFO] [mqtt_s3_multi_clients_comm_manager.py:306:send_message] mqtt_s3.send_message: to python client.
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [mqtt_s3_multi_clients_comm_manager.py:187:_on_message_impl] --------------------------
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [mqtt_s3_multi_clients_comm_manager.py:200:_on_message_impl] mqtt_s3.on_message: use s3 pack, s3 message key fedml_0_0_1_fb92d0a2-c8cf-4644-b7f6-8ec613987ba5
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [mqtt_s3_multi_clients_comm_manager.py:208:_on_message_impl] mqtt_s3.on_message: from python client.
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [mqtt_s3_multi_clients_comm_manager.py:211:_on_message_impl] mqtt_s3.on_message: model params length 2
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [mqtt_s3_multi_clients_comm_manager.py:182:_notify] mqtt_s3.notify: msg type = 2
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [client_master_manager.py:119:handle_message_receive_model_from_server] handle_message_receive_model_from_server.
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [client_master_manager.py:228:sync_process_group] sending round number to pg
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [client_master_manager.py:233:sync_process_group] round number 49 broadcasted to process group
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [client_master_manager.py:151:finish] Training finished for master client rank 0 in silo 0
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [client_slave_manager.py:43:await_sync_process_group] prcoess 1 received round_number 49
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [trainer_dist_adapter.py:133:cleanup_pg] Cleaningup process group for client 0 in silo 0
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [trainer_dist_adapter.py:133:cleanup_pg] Cleaningup process group for client 1 in silo 1
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [client_slave_manager.py:32:finish] Training finsihded for slave client rank 1 in silo 1
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [client_manager.py:129:finish] __finish client
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [mqtt_s3_multi_clients_comm_manager.py:337:stop_receive_message] mqtt_s3.stop_receive_message: stopping...
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [mqtt_s3_multi_clients_comm_manager.py:157:_on_disconnect] mqtt_s3.on_disconnect: disconnection returned result 0, user data None
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [client_master_manager.py:241:run] Connection is ready!
[FedML-Client(2) @device-id-1] [Sun, 01 May 2022 23:54:36] [INFO] [mqtt_s3_status_manager.py:78:_on_disconnect] mqtt_s3.on_disconnect: disconnection returned result 0, user data None
...
```

## Five lines of APIs

`torch_client.py`

```python
import fedml
from fedml.cross_silo.hierarchical import Client

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load_cross_silo(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    client = Client(args, device, dataset, model)
    client.run()
```

## Custom data and model 

The custom data and model example locates at the following folder:

`python/examples/cross_silo/mqtt_s3_fedavg_hierarchical_mnist_lr_example/custum_data_and_model`


```python
def load_data(args):
    n_dist_trainer = args.n_proc_in_silo
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

    # Split training data between distributed trainers
    train_data_local_dict = split_data_for_dist_trainers(train_data_local_dict, n_dist_trainer)

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