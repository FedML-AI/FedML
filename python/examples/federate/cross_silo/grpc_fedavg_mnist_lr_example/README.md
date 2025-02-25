# FedML Octopus Example with MNIST + Logistic Regression + gRPC Backend

FedML Octopus offers support for various communication backends. One of supported backends is gRPC. To use gRPC as backend your `comm_args` section of your config should match the following format:

```yaml
comm_args:
  backend: "GRPC"
  grpc_ipconfig_path: config/grpc_ipconfig.csv
```

`grpc_ipconfig_path` specifies the path of the config for gRPC communication. Config file specifies an ip address for each process through with they can communicate with each other. The config file should have the following format:

```csv
eid,rank,grpc_server_ip,grpc_server_port
0,0,0.0.0.0,8890
1,1,0.0.0.0,8899
2,2,0.0.0.0,8898
```

Here, `eid, rank, ip, port` are the id, rank, ip address and port of the server or client process. For server processes the rank is always set to 0, while for clients is always set to 1 or above.

## One Line API Example

Examples are provided at:

`python/examples/cross_silo/grpc_fedavg_mnist_lr_example/one_line`
`python/examples/cross_silo/grpc_fedavg_mnist_lr_example/step_by_step`
`python/examples/cross_silo/grpc_fedavg_mnist_lr_example/custom_data_and_model`

### Training Script

At the client side, the client ID (a.k.a rank) starts from 1.
Please also modify config/fedml_config.yaml, changing the `worker_num` the as the number of clients you plan to run.

At the server side, run the following script:
```
bash run_server.sh your_run_id
```

For client 1, run the following script:
```
bash run_client.sh 1 your_run_id
```
For client 2, run the following script:
```
bash run_client.sh 2 your_run_id
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