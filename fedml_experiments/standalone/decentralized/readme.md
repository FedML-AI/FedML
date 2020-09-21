# Central Server Free Federated Learning over Single-sided Trust Social Networks

We use this decentralized FL algorithm as an example to demonstrate how to develop decentralized ML algorithm based on FedML.

You can read the paper (https://arxiv.org/abs/1910.04956) to understand the algorithm used here.
No need to read the proof. Only understanding the algorithm will be enough to understand how to develop decentralized ML.
![](doc/readme_intro.jpg)

Please do not distribute when this is paper is under review.

## Requirements
Please refer to requirements.txt/environment.yml. You can rapidly build environment by using:
```
conda env create -f environment.yml

# or

pip install -r requirements.txt
```


## Datasets Preparation
all datasets are stored at "FedML/data" directory
```
# SUSY (FedML/data/UCI/SUSY)
sh download_SUSY_data.sh

# room occupancy (FedML/data/UCI/room_occupancy)
sh download_room_occupancy_data.sh
```

## Experiments Tracking Platform
We use [Weight and Bias](https://wandb.com) as our experiment management platform. 
Please refer its official documentation to setup your own environment.

```
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
```

## Software Architecture
|--main.py - entry point of our program; integrates the trainer.py and manages the parameters

|--trainer.py - main logic flow of the training; incooperates client.py, topology_manager.py

|--model.py - define our model using PyTorch

|--topology_manager.py - define the network

|--client_pushsum.py/client_dsgd.py - a node in the network; run the model and provides API to exchange with other clients

# How to Run
SUSY dataset:

```
#############################centralized online learning: (COL)###################################
# beta = 0, N = 128, undirected_neighbor_num=128, out_directed_neighbor_num=0
# sh run.sh "./../../../data/UCI/" 115 001 DOL 0.50 128 128 0 1
# 0.4782 (*)

#############################symmetric decentralized online learning (symmetric-DOL ###################################
# beta = 0, N = 128, undirected_neighbor_num=16, out_directed_neighbor_num=0
# sh run.sh "./../../../data/UCI/" 217 002 DOL 0.25 128 16 0 1
# 0.4910 (*)

############################# asymmetric decentralized online learning (asymmetric-DOL)###################
# beta = 0, N = 128, undirected_neighbor_num=16, out_directed_neighbor_num=16
# sh run.sh "./../../../data/UCI/" 314 003 DOL 0.50 128 16 16 0
# 0.4895 (*)

############################asymmetric pushsum-based decentralized online learning (asymmetric-pushsum)#################
# beta = 0, N = 128, undirected_neighbor_num=16, out_directed_neighbor_num=16
# sh run.sh "./../../../data/UCI/" 413 004 PUSHSUM 0.40 128 16 16 0
# 0.4822 (*)

#############################isolation online learning###################################
#beta = 0.5, N = 128, undirected_neighbor_num=128, out_directed_neighbor_num=0
#sh run.sh "./../../../data/UCI/" 1100 011 LOCAL 0.035 128 128 0 1
```
