## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Running Experiments 

```
nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 300 15 0.5 0 homo > ./mtl4.txt 2>&1 &

nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 300 15 0.1 0 homo > ./mtl1.txt 2>&1 &


nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 300 15 0.01 0 homo > ./mtl2.txt 2>&1 &

nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 300 15 0.001 0 homo > ./mtl3.txt 2>&1 &
nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 300 20 0.001 0 homo > ./mtl11.txt 2>&1 &




#######################MTL=1##############################
nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 300 15 0.5 1 homo > ./mtl22.txt 2>&1 &

nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 300 15 0.1 1 homo > ./mtl33.txt 2>&1 &


nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 300 15 0.01 1 homo > ./mtl44.txt 2>&1 &

nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 300 15 0.001 1 homo > ./mtl55.txt 2>&1 &
nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 300 20 0.001 1 homo > ./mtl66.txt 2>&1 &
```