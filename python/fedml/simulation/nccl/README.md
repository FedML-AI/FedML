AllReduce-based Federated Learning with Resource Scheduling

# Framework

Current FL simulating libraries support Single-Process or Muilt-Porcesses simulation. In Single-Process simulation, clients in FL will be sequentially simulated, limiting the simulating speed. In Muilt-Porcesses simulation, users must have GPUs as many as the number of clients per round or clients in total, excluding users who do not have a large GPU cluster.

In this simulating mode of FedML, users can maximally utilize their GPUs to simulate Federated Learning. The number of used GPUs can be customized. The number of clients per round and clients in total can be arbitrarily defined. 

We have three roles in our framework: Client, Server, LocalAggregator.

## Client
Each client, namely, is the user in federated learning. They will optimizie local models on their local datasets.

## Server
Server is responsible to 
1. aggregate models from all LocalAggregator and broadcast global model to all LocalAggregator. The communication between LocalAggregator and server is conducted by the NCCL acceleration.
2. schedule and assign different LocalAggregator (or GPUs) with different simulating tasks in each round. For example, in round 1, LocalAggregator 1 will simulate clients 1, 3, 5;  LocalAggregator 1 will simulate clients 6, 8.

## LocalAggregator
LocalAggregator (or a GPU) is responsible to
1. simulate the corresponding clients sequentially.
2. conduct local reduce of clients' models after simulating all clients assigned to it.
3. communicate local reduced model to the server and receive the global model from the server.




# Usage

Launch the NCCL simulation using the same cmd formats with the torch.distributed.launch.


python -m torch.distributed.launch \
    --nproc_per_node=5 --nnodes=1 --node_rank=0 \
    --master_addr localhost --master_port 11111 \
    torch_fedavg_mnist_lr_custum_data_and_model_example.py --cf config/fedml_config.yaml

Please refer to example/simulation/nccl_fedavg_example for more examples



















