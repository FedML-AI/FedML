# FedCV - Object Detection

## Prerequisites & Installation

```bash
pip install fedml --upgrade
```

There are other dependencies in some tasks that need to be installed.

```bash
git clone https://github.com/FedML-AI/FedML
cd FedML/python/app/fedcv/object_detection

cd config/
bash bootstrap.sh

cd ..
```

### Run the MPI simulation

```bash
bash run_simulation.sh [CLIENT_NUM]
```

To customize the number of client, you can change the following variables in `config/simulation/fedml_config.yaml`:

```bash
train_args:
  federated_optimizer: "FedAvg"
  client_id_list:
  client_num_in_total: 2 # change here!
  client_num_per_round: 1 # change here!
  comm_round: 20
  epochs: 5
  batch_size: 1
```

### Run the server and client using MQTT

If you want to run the edge server and client using MQTT, you need to run the following commands.

```bash
bash run_server.sh

# in a new terminal window

# run the client 1
bash run_client.sh 1

# run the client with client_id
bash run_client.sh [CLIENT_ID]
```

To customize the number of client, you can change the following variables in `config/fedml_config.yaml`:

```bash
train_args:
  federated_optimizer: "FedAvg"
  client_id_list:
  client_num_in_total: 2 # change here!
  client_num_per_round: 2 # change here!
  comm_round: 20
  epochs: 5
  batch_size: 1
```

### Run the application using MLOps

You just need to select the YOLOv5 Object Detection application and start a new run.

Run the following command to login to MLOps.

```bash
fedml login [ACCOUNT_ID]
```

### Build your own application

1. Build package

```bash
pip install fedml --upgrade
bash build_mlops_pkg.sh
```

2. Create an application and upload package in mlops folder to MLOps

## Change model

The default model is YOLOv5. You can change the model by replacing the `config/fedml_config.yaml` file with `config/fedml_config_yolovx.yaml`.

Or you can change the model by replacing the `model` and `yolo_cfg` in `config/fedml_config.yaml` with your own model and configuration.
