## Training Script

Pleaes add hostname/IP of the server and clients to config/mpi_host_file according to the following format:

```
localhost
node1
node2
```

Then, run the following script, where client_num is the number of clients you wish to train.
```
bash run_server.sh $client_num
```

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