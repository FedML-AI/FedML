# FedML Octopus Examples

## Horizontal Federated Learning

- [mqtt_s3_fedavg_mnist_lr_example](./examples/mqtt_s3_fedavg_mnist_lr_example.md): an example to illustrate how to run horizontal federated learning in data silos (hospitals, banks, etc.)

## Hierarchical Federated Learning

- [hierarchical_fedavg_mnist_lr_example](./examples/mqtt_s3_fedavg_hierarchical_mnist_lr_example.md): an example to illustrate how to run hierarchical federated learning in data silos (hospitals, banks, etc.). 
As shown in the figure below, here `hierarchical` means that inside each FL Client (data silo), there are multiple GPUs which can run local distributed training with PyTorch DDP, and then the FL server aggregates globally from the results received from all FL Clients. 

<img src="./../_static/image/cross-silo-hi.png" alt="parrot" style="width:100%;"/>

