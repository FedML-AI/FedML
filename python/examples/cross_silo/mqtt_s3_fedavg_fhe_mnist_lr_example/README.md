## Training Script
FHE-based FedAvg: using CKKS scheme from PALISADE library. Implementation can be found under `python/fedml/core/fhe`.

To run: 

Please modify config/fedml_config.yaml, changing the `client_num_in_total` the as the number of clients you plan to run.

Note: please run the server first.


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