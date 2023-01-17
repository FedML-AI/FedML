## Training Script


Then, run the following script, where client_num is the number of clients you wish to train.
```
bash run_cross_silo_mpi.sh $client_num
```

If you need to configure GPU device mapping, please refer to `config/gpu_mapping.yaml`.

Please note the node running the script should have passwordless SSH access to other nodes.

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