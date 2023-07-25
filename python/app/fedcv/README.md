# FedCV - Object Detection

## Prerequisites & Installation

```bash
pip install fedml --upgrade
```

## Prepare YOLOv6

Download the YOLOv6-S6 checkpoint from `https://github.com/meituan/YOLOv6` and add the checkpoint path to `./YOLOv6/configs/yolov6s6_finetune.py`.

## Prepare VOC dataset
Download the VOC dataset from `https://yolov6-docs.readthedocs.io/zh_CN/latest/%E5%85%A8%E6%B5%81%E7%A8%8B%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97/%E8%AE%AD%E7%BB%83%E8%AF%84%E4%BC%B0%E6%8E%A8%E7%90%86%E6%B5%81%E7%A8%8B.html#id2` and run `python ./YOLOv6/yolov6/data/voc2yolo.py --voc_path your_path/to/VOCdevkit`. Then, fill in the path in `./YOLOv6/data/voc.yaml`.

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
  client_num_per_round: 2 # change here!
  comm_round: 10000
  epochs: 1
  steps: 8
  batch_size: 8
```

### Run the server and client using MQTT

If you want to run the edge server and client using MQTT, you need to run the following commands.

> !!IMPORTANT!! In order to avoid crosstalk during use, it is strongly recommended to modify `run_id` in `run_server.sh` and `run_client.sh` to avoid conflict.

```bash
bash run_server.sh your_run_id

# in a new terminal window

# run the client 1
bash run_client.sh [CLIENT_ID] your_run_id

# run the client with client_id
bash run_client.sh [CLIENT_ID] your_run_id
```

### Build your own application

1. Build package

```bash
pip install fedml --upgrade
bash build_mlops_pkg.sh
```

2. Create an application and upload package in mlops folder to MLOps
