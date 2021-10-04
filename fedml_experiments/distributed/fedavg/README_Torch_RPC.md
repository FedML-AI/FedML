## Performance Test for Torch RPC in Cross-silo Federated Learning


### 1. Prepare the dataset 
Please execute the following command in each node:
```
USER=chaoyanghe
FedML_WORKSPACE=/home/$USER/FedML
cd $/data/MNIST
sh download_and_unzip.sh 
```

### 2. Torch RPC configuration
* configure `mpi_host_file`.

Currently, we use MPI4Py to launch multiple processes across multiple nodes (servers).
We will remove the dependency of MPI4Py later when using torch RPC.

* configure `gpu_mapping.yaml`
```
# When there are 10 clients and 1 FL server, the following mapping means that in the first server, we assign 6 processes, including one for the server, and the 5 others for the clients.
and in the second server, we assign 5 processes (clients) to the first 4 GPUs.
mapping_FedML_tRPC:
    lambda-server1: [0, 0, 0, 0, 2, 2, 1, 1]
    lambda-server2: [2, 1, 1, 1, 0, 0, 0, 0]
```


### Training Scripts
``` training
FedML_WORKSPACE=/home/$USER/FedML
cd $FedML_WORKSPACE/fedml_experiments/distributed/fedavg

# kill processes
kill $(ps aux | grep "main_fedavg.py" | grep -v grep | awk '{print $2}')
```