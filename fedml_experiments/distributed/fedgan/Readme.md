## Abstract
This is the toy example for the federated learning based gan. 


FedGAN Api is created by Lei Gao, Tuo Zhang, Qi Chang ang Zhennan. For any questions during the using, please contact us via the FedML Slack.

## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments
The toy examlpes is training a GAN with the MNIST dataset.
Note please tune hyper-parameters accordingly. 
You can refer the hyper-parameter suggestions at `FedML/benchmark/README.md`), but this may not be the optimal.

## Usage
```
sh run_fedavg_distributed_pytorch.sh \
    --client_num_in_total <client_num_in_total> \
    --client_num_per_round <client_num_per_round> \
    --model <model> \
    --partition_method <partition_method> 
    --comm_round <comm_round> \
    --epochs <epochs>\
    --batch_size <batch_size> \
    --learning_rate <learning_rate> \
    --dataset <dataset> \
    --data_dir <data_dir> \
    --client_optimizer <client_optimizer> \
    --backend <backend> \
    --grpc_ipconfig_path <grpc_ipconfig_path> \
    --ci <ci>
```
