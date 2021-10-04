## Performance Test for Torch RPC in Cross-silo Federated Learning


### Prepare the dataset 
Please execute the following command in each node:
```
USER=chaoyanghe
FedML_WORKSPACE=/home/$USER/FedML_tRPC
cd $/data/MNIST
sh download_and_unzip.sh 
```

### Training Scripts
``` training
FedML_WORKSPACE=/home/$USER/FedML_tRPC

# run the client first

cd $FedML_WORKSPACE/fedml_experiments/distributed/fedavg
sh run_fedavg_cross_zone.sh 1
sh run_fedavg_cross_zone.sh 2

# after all clients are up, run the server 
# this order is important because server needs to build connection according to active clients.
# we will solve this contraints by developing a shaking hand protocol.

cd $FedML_WORKSPACE/fedml_experiments/distributed/fedavg
sh run_fedavg_cross_zone.sh 0


# kill processes
kill $(ps aux | grep "main_fedavg.py" | grep -v grep | awk '{print $2}')
```