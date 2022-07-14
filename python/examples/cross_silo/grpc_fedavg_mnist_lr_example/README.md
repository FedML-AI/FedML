# FedML Octopus Example with MNIST + Logistic Regression + gRPC Backend

FedML Octopus offers support for various communication backends. One of supported backends is gRPC. To use gRPC as backend your `comm_args` section of your config should match the following format:

```yaml
comm_args:
  backend: "GRPC"
  grpc_ipconfig_path: config/grpc_ipconfig.csv
```

`grpc_ipconfig_path` specifies the path of the config for gRPC communication. Config file specifies an ip address for each process through with they can communicate with each other. The config file should have the folliwng format:

```csv
receiver_id,ip
0,127.0.0.1
1,127.0.0.1
2,127.0.0.1
```

Here the `receiver_id` is the rank of the process.

## One Line API Example

Example is provided at:

`python/examples/cross_silo/grpc_fedavg_mnist_lr_example/one_line`
### Training Script

At the client side, the client ID (a.k.a rank) starts from 1.
Please also modify config/fedml_config.yaml, changing the `worker_num` the as the number of clients you plan to run.

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


The step by step example using five lines of code locates at the following folder:

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