## Training Script

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
Note: 
1. please change the run_id in run_client/run_server scripts to your own.
2. For Windows users, please use *.bat scripts.

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