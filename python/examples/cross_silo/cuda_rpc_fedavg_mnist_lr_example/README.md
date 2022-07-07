# FedML Octopus Example with MNIST + Logistic Regression + TRPC + CUDA RPC

FedML Octopus offers support for various communication backends. If you opt for using TRPC backend you can take advantege of CUDA RPC for direct GPU to GPU tranfer. To use gRPC as backend your `comm_args` section of your config should match the following format:

```yaml
device_args:
  using_gpu: false
  ...

comm_args:
  backend: "TRPC"
  trpc_master_config_path: config/trpc_master_config.csv
  enable_cuda_rpc: True
  cuda_rpc_gpu_mapping:
    # Rank: GPU_index
    0: 0
    1: 2
    2: 4
    ...
```

For info on `trpc_master_config_path` refer to `python/examples/cross_silo/cuda_rpc_fedavg_mnist_lr_example/one_line`.
`cuda_rpc_gpu_mapping` should map each process rank to it's corresponding device id. For example in this example process with rank 0 (server) uses GPU 1, process with rank 1 uses GPU 2 and process with rank 2 uses GPU 4.

## One Line API Example

Example is provided at:

`python/examples/cross_silo/cuda_rpc_fedavg_mnist_lr_example/one_line`
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