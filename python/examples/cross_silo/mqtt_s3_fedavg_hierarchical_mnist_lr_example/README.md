## Training Script

At the client side, the client ID (a.k.a rank) starts from 1, and NODE_RANK for each silo starts from 0.
Please also modify config/fedml_config.yaml, changing the `worker_num` the as the number of clients you plan to run.

At the server side, run the following script:
```
bash run_server.sh 
```

For Silo/Client 1, run the following script on the first node (NODE_RANK=0):
```
bash run_client.sh 1 0
```
For Silo/Client 1, run the following script on the second node (NODE_RANK=1):
```
bash run_client.sh 1 1
```

For Silo/Client 2, run the following script:
```
bash run_client.sh 2 0
```
Note: please run the server first.
```
Note: please run the server first.

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