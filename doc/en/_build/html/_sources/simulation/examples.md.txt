# FedML Parrot Examples

## Simulation with a Single Process (Standalone)

- [sp_fedavg_mnist_lr_example](./examples/sp_fedavg_mnist_lr_example.md): 
  Simulating FL using a single process in your personal laptop or server. This is helpful when researchers hope to try a quick algorithmic idea in small synthetic datasets (MNIST, shakespeare, etc.) and small models (ResNet-18, Logistic Regression, etc.). 

## Simulation with Message Passing Interface (MPI)
- [mpi_torch_fedavg_mnist_lr_example](./examples/mpi_torch_fedavg_mnist_lr_example.md): 
  MPI-based Federated Learning for cross-GPU/CPU servers.
  

## Simulation with NCCL-based MPI (the fastest training)
- In case your cross-GPU bandwidth is high (e.g., InfiniBand, NVLink, EFA, etc.), we suggest to use this NCCL-based MPI FL simulator to accelerate your development. 
