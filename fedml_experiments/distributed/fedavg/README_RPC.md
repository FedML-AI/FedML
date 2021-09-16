## Cross-AWS Zone (cross-silo) Federated Learning System

### login permissions
```
cp grpc-fedml.pem ~/.ssh/grpc-fedml.pem
vim ~/.ssh/config

# add your servers to ~/.ssh/config. Here we have two nodes:
Host fedml-gprc-1
  hostname ec2-54-188-191-111.us-west-2.compute.amazonaws.com
  user ec2-user
  IdentityFile ~/.ssh/grpc-fedml.pem
  
Host fedml-gprc-2
  hostname ec2-18-237-130-223.us-west-2.compute.amazonaws.com
  user ec2-user
  IdentityFile ~/.ssh/grpc-fedml.pem
```

### Sync source code ("async_fedml_code_node1.sh", "async_fedml_code_node2.sh")
``` 
#!/bin/bash
DEV_NODE=fedml-grpc-1
LOCAL_PATH=/Users/hchaoyan/source/FedML
REMOTE_PATH=/home/ec2-user/FedML_gRPC
alias ws-sync='rsync -avP -e ssh --exclude '.idea' $LOCAL_PATH $DEV_NODE:$REMOTE_PATH --progress --append'
ws-sync; fswatch -o $LOCAL_PATH | while read f; do ws-sync; done
```

```
#!/bin/bash
DEV_NODE=fedml-grpc-2
LOCAL_PATH=/Users/hchaoyan/source/FedML
REMOTE_PATH=/home/ec2-user/FedML_gRPC
alias ws-sync='rsync -avP -e ssh --exclude '.idea' $LOCAL_PATH $DEV_NODE:$REMOTE_PATH --progress --append'
ws-sync; fswatch -o $LOCAL_PATH | while read f; do ws-sync; done
```

### Prepare the dataset 
Please execute the following command in each node:
```
FedML_WORKSPACE=/home/ec2-user/FedML_gRPC
cd $/data/MNIST
sh download_and_unzip.sh 
```

### Training Scripts
``` training
FedML_WORKSPACE=/home/ec2-user/FedML_gRPC

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