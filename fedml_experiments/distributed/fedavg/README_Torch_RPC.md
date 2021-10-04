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

Note that you may need to set SSH mutual trust for the master node:
```
# node master
cd /home/$USER/.ssh
ssh-keygen

# other nodes, copy the string in /home/$USER/.ssh/id_rsa.pub to 
vim /home/$USER/.ssh/authorized_keys
```

* configure `gpu_mapping.yaml`
```
# When there are 10 clients and 1 FL server, the following mapping means that in the first server, we assign 6 processes, including one for the server, and the 5 others for the clients.
and in the second server, we assign 5 processes (clients) to the first 4 GPUs.
mapping_FedML_tRPC:
    lambda-server1: [0, 0, 0, 0, 2, 2, 1, 1]
    lambda-server2: [2, 1, 1, 1, 0, 0, 0, 0]
```

* configure `trpc_master_config.csv`
```
master_ip, master_port
192.168.11.1, 29500
```

* configure network inferface, check `run_fedavg_trpc.sh` for details:
```
# enable InfiniBand
#export NCCL_SOCKET_IFNAME=ib0
#export GLOO_SOCKET_IFNAME=ib0
#export TP_SOCKET_IFNAME=ib0
#export NCCL_IB_HCA=ib0

# disable InfiniBand
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eno2
export GLOO_SOCKET_IFNAME=eno2
export TP_SOCKET_IFNAME=eno2

```


### Training Scripts
``` training
FedML_WORKSPACE=/home/$USER/FedML
cd $FedML_WORKSPACE/fedml_experiments/distributed/fedavg
nohup sh run_fedavg_trpc.sh > run_fedavg_trpc.log 2>&1 &

# follow the log
tail -f run_fedavg_trpc.log

# search error in log
vim run_fedavg_trpc.log

# kill processes
kill $(ps aux | grep "main_fedavg.py" | grep -v grep | awk '{print $2}')
```