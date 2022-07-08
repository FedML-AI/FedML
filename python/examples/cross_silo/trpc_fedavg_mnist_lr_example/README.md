# FedML Octopus Example with MNIST + Logistic Regression + Torch RPC Backend

FedML Octopus offers support for various communication backends. One of supported backends is Troch RPC (TRPC). To use TRPC as backend your `comm_args` section of your config should match the following format.

```yaml
comm_args:
  backend: "TRPC"
  trpc_master_config_path: config/trpc_master_config.csv
```

`trpc_master_config_path` specifies the path of the config for Torch RPC master process which is the same as server process in FedML. Config file specifies an ip and port which will be used by other processes to communicate with master process and should have the folliwng format:

```csv
master_ip, master_port
127.0.0.1,29600
```


## One Line API Example

Example is provided at:

`python/examples/cross_silo/trpc_fedavg_mnist_lr_example/one_line`
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