## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments
Note please tune hyper-parameters accordingly. 
You can refer the hyper-parameter suggestions at `FedML/benchmark/README.md`), but this may not be the optimal.


## Usage
```
sh run_fedavg_corss_silo.sh \
  --model N             neural network used in training (default: mobilenet)
  --dataset N           dataset used for training (default: cifar10)
  --data_dir DATA_DIR   data directory (default: ./../../../data/cifar10)
  --partition_method N  how to partition the dataset on local workers
                        (default: hetero)
  --partition_alpha PA  partition alpha (default: 0.5) (default: 0.5)
  --client_silo_num_in_total NN
                        number of workers in a distributed cluster (default:
                        1000)
  --silo_num_per_round NN
                        number of workers (default: 4)
  --batch_size N        input batch size for training (default: 64) (default:
                        64)
  --client_optimizer CLIENT_OPTIMIZER
                        SGD with momentum; adam (default: adam)
  --backend BACKEND     Backend for Server and Client (default: MPI)
  --lr LR               learning rate (default: 0.001) (default: 0.001)
  --wd WD               weight decay parameter; (default: 0.0001)
  --epochs EP           how many epochs will be trained locally (default: 5)
  --comm_round COMM_ROUND
                        how many round of communications we shoud use
                        (default: 10)
  --is_mobile IS_MOBILE
                        whether the program is running on the FedML-Mobile
                        server side (default: 1)
  --frequency_of_the_test FREQUENCY_OF_THE_TEST
                        the frequency of the algorithms (default: 1)
  --gpu_server_num GPU_SERVER_NUM
                        gpu_server_num (default: 1)
  --gpu_num_per_server GPU_NUM_PER_SERVER
                        gpu_num_per_server (default: 4)
  --gpu_mapping_file GPU_MAPPING_FILE
                        the gpu utilization file for servers and clients. If
                        there is no gpu_util_file, gpu will not be used.
                        (default: None)
  --gpu_mapping_key GPU_MAPPING_KEY
                        the key in gpu utilization file (default:
                        mapping_default)
  --grpc_ipconfig_path GRPC_IPCONFIG_PATH
                        config table containing ipv4 address of grpc server
                        (default: grpc_ipconfig.csv)
  --trpc_master_config_path TRPC_MASTER_CONFIG_PATH
                        config indicating ip address and port of the master
                        (rank 0) node (default: trpc_master_config.csv)
  --enable_cuda_rpc     Enable cuda rpc (only for TRPC backend) (default:
                        False)
  --silo_node_rank SILO_NODE_RANK
                        rank of the node in silo (default: 0)
  --silo_rank SILO_RANK
                        rank of the silo (default: 0)
  --nnode NNODE         number of nodes in silo (default: 1)
  --nproc_per_node NPROC_PER_NODE
                        number of processes in each node (default: 1)
  --pg_master_address PG_MASTER_ADDRESS
                        address of the DDP process group master (default: 1)
  --pg_master_port PG_MASTER_PORT
                        port of the DDP process group master (default: 1)
  --silo_gpu_mapping_file SILO_GPU_MAPPING_FILE
                        the gpu utilization file for silo processes. (default:
                        None)
  --mqtt_config_path MQTT_CONFIG_PATH
                        Path of config for mqtt server. (default: None)
  --s3_config_path S3_CONFIG_PATH
                        Path of config for S3 server. (default: None)
  --ci CI               CI (default: 0)
```



## Cross-Silo Setting

Cross-Silo setting provides the possibility of running algorithms in a heterogeneous and hierarchical environment. Each silo will specify its architecture using the following arguments (each node of the silo should run the script with appropriate arguments).

**--silo_rank**

This argument specifies the rank of the silo among other silos starting from 0. Please note that the silo with rank 0 will be used as FedML server (aggregator).

**--nnode**

This argument specifies number of nodes in the silo.

**--nproc_per_node**

This argument specifies number of processes per node. Each process can be assigned to a GPU.

**--silo_node_rank**

This argument specifies rank of the node in the silo starting from 0.

**--pg_master_address and --pg_master_port**

These arguments specify IP address and port of the master of process group. These arguments will be used by the processes in the silo to participate in the distributed training by joining the process group. --pg_master_address should receive the IP of the node with rank 0.

**--silo_gpu_mapping_file**

This argument specifies the path of the GPU config for silo. 
Example silo GPU config mapping file:
```
server-1: [0,0,0,0,1,1,0,0]
server-2: [1,1,0,0,0,0,0,0]
```
In this example, the silo has two nodes (server-1 and server-2). Each node has two processes and 8 GPU. On server-1, GPUs 5 and 6 will be assigned to the processes and on server-1, GPUs 1 and 2 will be assigned to the processes.

### Process rank
Each process in the silo will be assigned a process_rank under the hood. Processes in node with rank $i$ will be assigned ranks  $i \times \text{nproc\_per\_node}...(i \times \text{nproc\_per\_node} - 1)$

### --gpu_mapping_file in Cross-Silo Setting
Please note in this setting `--gpu_mapping_file` specifies the path of GPU mapping file for inter-silo messaging passing which will be used only if the backend arguments is set as `--backend TRPC` and `--enable-cuda-rpc` is passed alongside it. In this case, the mapping should reflect the GPU of rank 0 process in each silo.

Assume we have a topology that contains a central server (FL server), and two FL clients (silos), we only the process 0 in each silo to do PyTorch RPC communication. So the example configufation is as follows:

```
mapping_FedML_cross_silo:
    FL-Server: [0,0,0,1,0,0,0,0]
    silo1-FL-Client: [0,0,0,0,1,0,0,0]
    silo2-FL-Client: [0,0,0,0,1,0,0,0]
```
In this example, the FL Server specifies GPU 4 and the two FL silos specifies GPU 5 to be used in the messaging passing between the client and the server using CUDA RPC of PyTorch.


## Setting ip configurations for grpc
```

1. create .csv file in the format:

    receiver_id,ip
    0,<ip_0>
    ...
    n,<ip_n>
    
    where n = client_num_per_round

2. provide path to file as argument to --grpc_ipconfig_path
```

## Setting configurations for MQTT

In order to use MQTT or MQTT_S3 as backend, you need to provide configuration for MQTT broker as follows:

```
1. create .yaml file in the format:

    BROKER_HOST: <broker_ip>
    BROKER_PORT: <broker_port>
    MQTT_USER: <username>
    MQTT_PWD: <password>

2. provide path to file as argument to --mqtt_config_path
```

## Setting configurations for S3

In order to use MQTT_S3 as backend, you need to provide configuration for S3 as follows:

```

1. create .yaml file in the format:

    BUCKET_NAME: <bucket_name>
    CN_S3_AKI: <access_key_id>
    CN_S3_SAK: <secret_access_key>
    CN_REGION_NAME: <region>

2. provide path to file as argument to --s3_config_path
```



## Setting ip configurations for grpc
```

1. create .csv file in the format:

    receiver_id,ip
    0,<ip_0>
    ...
    n,<ip_n>
    
    where n = client_num_per_round

2. provide path to file as argument to --grpc_ipconfig_path
```

## Running using TRPC
In order to run using TRPC set master's address and port in file trpc_master_config.csv, and use TRPC as backend option.

## Experiments
```Configuration
Central Server: AWS Server (IP:Port)
Hierarchical topology:
client 1: Lambda 1 & 2 (2 nodes, each node has 8 GPUs -> DDP)
client 2: Lambda 4 (1 nodes, CPUs)
client 3: Chaoyang's Personal GPU Server (1 node, 4 GPUs)

Communication backend: MQTT + S3

FL Algorithm: FedAvg

Sync or Async: Async
```

```Scripts

# run client
cd fedml_experiments/distributed/fedavg_cross_silo
sh run_client.sh

# run server
cd fedml_experiments/distributed/fedavg_cross_silo
sh run_server.sh


```

