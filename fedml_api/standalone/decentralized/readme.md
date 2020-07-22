# Central Server Free Federated Learning over Single-sided Trust Social Networks
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
```
# SUSY
sh ./data/download_SUSY_data.sh

# room occupancy
sh ./data/download_room_occupancy_data.sh
```

## Experiments Tracking Platform
We use [Weight and Bias](https://wandb.com) as our experiment management platform. 
Please refer its official documentation to setup your own environment.

If you don't want this functionality, please change the code in the main.py file:
```
switch_wandb = False
```

## Software Architecture
|--main.py - entry point of our program; integrates the trainer.py and manages the parameters

|--trainer.py - main logic flow of the training; incooperates client.py, topology_manager.py

|--model.py - define our model using PyTorch

|--topology_manager.py - define the network

|--client_pushsum.py/client_dsgd.py - a node in the network; run the model and provides API to exchange with other clients

# How to Run
Please refer to "run.sh" script. 

