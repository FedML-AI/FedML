# FedML Octopus Example with MNIST + Logistic Regression + Torch RPC Backend

FedML Octopus offers support for various communication backends. One of supported backends is MPI. To use TRPC as backend your `comm_args` section of your config should match the following format.

```yaml
comm_args:
  backend: "MPI"
```
Pleaes add hostname/IP of the server and clients to config/mpi_host_file according to the following format:

```
localhost
node1
node2
```

Please note the node running the scripts should have passwordless SSH access to other nodes.

## One Line API Example

Example is provided at:

`python/examples/cross_silo/mpi_fedavg_mnist_lr_example/one_line`

## Training Script

After adding the hostnames to config/mpi_host_file, run the following script, where client_num is the number of clients you wish to train.
```
bash run_server.sh $client_num
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