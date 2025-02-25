
## Prerequisites
At the client side, the client ID (a.k.a rank) starts from 1. 
Please also modify `config/fedml_config.yaml` as you see fit. Changing the `worker_num` the as the number of clients you plan to run.
The default ip of every groc server is set to `0.0.0.0`, and all grpc ports start from 8890 and increase based on the client's rank.

> **_NOTE:_** 
> The `config/grpc_ipconfig.csv` file contains only one record referring to the grpc server of 
> the aggregator (rank: 0). This record is mandatory. However, you can change the values of the `ip` and `port` 
> attributes as you see fit, and more records for grpc server of the rest of clients. For instance:
```
eid,rank,grpc_server_ip,grpc_server_port
0,0,0.0.0.0,8890
1,1,0.0.0.0,8891
2,2,0.0.0.0,8892
```

## Start Script

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

## A Better User-experience with FedML FLOps (fedml.ai)
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