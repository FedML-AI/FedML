# FedML Octopus Example with MNIST + Logistic Regression

This example illustrates how to do real-world cross-silo federated learning with FedML Octopus. The source code locates at [https://github.com/FedML-AI/FedML/tree/master/python/examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example](https://github.com/FedML-AI/FedML/tree/master/python/examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example).

## One line API

`python/examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example/one_line`

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
Note: please run the server first.



`config/fedml_config.yaml` is shown below.

Here `common_args.training_type` is "cross_silo" type, and `train_args.client_id_list` needs to correspond to the client id in the client command line.

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
```

### Training Results

At the end of the 50th training round, the server window will see the following output:

```shell
FedML-Server(0) @device-id-0 - Wed, 27 Apr 2022 03:38:28 fedml_aggregator.py[line:199] INFO ################test_on_server_for_all_clients : 49
FedML-Server(0) @device-id-0 - Wed, 27 Apr 2022 03:38:38 fedml_aggregator.py[line:225] INFO {'training_acc': 0.7155714841722886, 'training_loss': 1.8997359397010631}
FedML-Server(0) @device-id-0 - Wed, 27 Apr 2022 03:38:38 mlops_metrics.py[line:67] INFO report_server_training_metric. message_json = {'run_id': '0', 'round_idx': 49, 'timestamp': 1651030718.803107, 'accuracy': 0.7156, 'loss': 1.8997}
FedML-Server(0) @device-id-0 - Wed, 27 Apr 2022 03:38:40 fedml_aggregator.py[line:262] INFO {'test_acc': 0.717948717948718, 'test_loss': 1.8972983557921448}
FedML-Server(0) @device-id-0 - Wed, 27 Apr 2022 03:38:40 mlops_metrics.py[line:74] INFO report_server_training_round_info. message_json = {'run_id': '0', 'round_index': 49, 'total_rounds': 50, 'running_time': 930.5741}
...
```

At the end of the 50th training round, the client1 window will see the following output:

```shell
FedML-Client(1) @device-id-1 - Wed, 27 Apr 2022 03:38:45 fedml_client_manager.py[line:145] INFO #######training########### round_id = 50
[2022-04-27 03:38:45,591] [INFO] [my_model_trainer_classification.py:56:train] Update Epoch: 0 [10/20 (50%)]    Loss: 1.984373
[2022-04-27 03:38:45,602] [INFO] [my_model_trainer_classification.py:56:train] Update Epoch: 0 [20/20 (100%)]   Loss: 2.053363
[2022-04-27 03:38:45,602] [INFO] [my_model_trainer_classification.py:63:train] Client Index = 1 Epoch: 0        Loss: 2.018868
FedML-Client(1) @device-id-1 - Wed, 27 Apr 2022 03:38:45 client_manager.py[line:107] INFO Sending message (type 3) to server
FedML-Client(1) @device-id-1 - Wed, 27 Apr 2022 03:38:45 mqtt_s3_multi_clients_comm_manager.py[line:240] INFO mqtt_s3.send_message: starting...
FedML-Client(1) @device-id-1 - Wed, 27 Apr 2022 03:38:45 mqtt_s3_multi_clients_comm_manager.py[line:296] INFO mqtt_s3.send_message: S3+MQTT msg sent, message_key = fedml_0_1_06180ace-1d4b-445c-b3d7-4ae765659471
FedML-Client(1) @device-id-1 - Wed, 27 Apr 2022 03:38:45 mqtt_s3_multi_clients_comm_manager.py[line:306] INFO mqtt_s3.send_message: to python client.
FedML-Client(1) @device-id-1 - Wed, 27 Apr 2022 03:38:47 mlops_metrics.py[line:81] INFO report_client_model_info. message_json = {'run_id': '0', 'edge_id': 1, 'round_idx': 51, 'client_model_s3_address': '...'}
...
```

At the end of the 50th training round, the client2 window will see the following output:


```shell
FedML-Client(2) @device-id-1 - Wed, 27 Apr 2022 03:38:58 fedml_client_manager.py[line:145] INFO #######training########### round_id = 50
[2022-04-27 03:38:58,128] [INFO] [my_model_trainer_classification.py:56:train] Update Epoch: 0 [10/20 (50%)]    Loss: 1.984373
[2022-04-27 03:38:58,137] [INFO] [my_model_trainer_classification.py:56:train] Update Epoch: 0 [20/20 (100%)]   Loss: 2.053363
[2022-04-27 03:38:58,137] [INFO] [my_model_trainer_classification.py:63:train] Client Index = 2 Epoch: 0        Loss: 2.018868
FedML-Client(2) @device-id-1 - Wed, 27 Apr 2022 03:38:58 client_manager.py[line:107] INFO Sending message (type 3) to server
FedML-Client(2) @device-id-1 - Wed, 27 Apr 2022 03:38:58 mqtt_s3_multi_clients_comm_manager.py[line:240] INFO mqtt_s3.send_message: starting...
FedML-Client(2) @device-id-1 - Wed, 27 Apr 2022 03:38:58 mqtt_s3_multi_clients_comm_manager.py[line:296] INFO mqtt_s3.send_message: S3+MQTT msg sent, message_key = fedml_0_2_a33a50ad-c289-41b8-a925-9205eea272f2
FedML-Client(2) @device-id-1 - Wed, 27 Apr 2022 03:38:58 mqtt_s3_multi_clients_comm_manager.py[line:306] INFO mqtt_s3.send_message: to python client.
FedML-Client(2) @device-id-1 - Wed, 27 Apr 2022 03:38:58 mlops_metrics.py[line:81] INFO report_client_model_info. message_json = {'run_id': '0', 'edge_id': 2, 'round_idx': 51, 'client_model_s3_address': '...'}
...
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

`python/examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example/custum_data_and_model`


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