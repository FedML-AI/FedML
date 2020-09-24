## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Running Experiments 

```
nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 500 15 0.1 0 > ./mtl1.txt 2>&1 &
nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 500 1 0.1 0 > ./mtl3.txt 2>&1 &


nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 500 15 0.01 0 > ./mtl2.txt 2>&1 &
nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 500 1 0.01 0 > ./mtl2.txt 2>&1 &

nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 500 15 0.001 0 > ./mtl3.txt 2>&1 &
nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 500 1 0.001 0 > ./mtl3.txt 2>&1 &
nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 500 20 0.001 0 homo > ./mtl11.txt 2>&1 &



#######################MTL=1##############################
nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 500 15 0.1 1 > ./mtl4.txt 2>&1 &
nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 500 1 0.1 1 > ./mtl4.txt 2>&1 &

nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 500 15 0.01 1 > ./mtl4.txt 2>&1 &
nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 500 1 0.01 1 > ./mtl4.txt 2>&1 &

nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 500 15 0.001 1 > ./mtl4.txt 2>&1 &
nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 500 1 0.001 1 > ./mtl4.txt 2>&1 &
nohup sh run_decentralized_mtl_distributed_pytorch.sh 8 8 500 20 0.001 1 homo > ./mtl4.txt 2>&1 &
```