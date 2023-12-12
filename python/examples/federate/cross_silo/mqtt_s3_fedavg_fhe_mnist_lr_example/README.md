## Training Script
HE-based FedAvg: example using CKKS scheme from TenSeal. Implementation can be found under `python/fedml/core/fhe`.

## Setup

Install TenSeal library
 ```
 pip3 install tenseal
 ```



## After installation
(Optional) Please modify config/fedml_config.yaml, changing the `client_num_in_total` the as the number of clients (default: 2) you plan to run.

To run clients, open more terminal windows to spawn clients at will:

go to the example for all terminals (including the server):
```
cd python/examples/cross_silo/mqtt_s3_fedavg_fhe_mnist_lr_example
```
Note: make sure there is a context.pickle file (crypto context) under this dir

Note: please run the server first!

At the server side, run the following script:
```
bash run_server.sh your_run_id
```
At the client side, the client ID (a.k.a rank) starts from 1. For an example with 2 clients:

At Client 1, run the following script:
```
bash run_client.sh 1 your_run_id
```
At Client 2, run the following script:
```
bash run_client.sh 2 your_run_id
```

## Key Management

Current implementation saves cryptocontext and keys in the local folder. To deploy the system, a key authority is needed to generate and distribute both cryptocontext and keys. Cryptocontext will be sent to both clients and the server. Keys will be sent to only clients. 

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

## Acknowledgement
We use the TenSeal library for basic building CKKS blocks.