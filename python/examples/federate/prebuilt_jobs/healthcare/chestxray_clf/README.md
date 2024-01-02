# ChestXRay Classification

## Introduction

WIP.

## Prerequisites & Installation

```bash
pip install fedml --upgrade
```

There are other dependencies in some tasks that need to be installed.

```bash
git clone https://github.com/FedML-AI/FedML
cd FedML/python/app/healthcare/chestxray_clf/

cd config/
bash bootstrap.sh

cd ..

# Download data
cd data/chestxray/
bash download_and_unzip.sh
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
  client_num_per_round: 2 # change here!
  comm_round: 20
  epochs: 5
  batch_size: 1
```

### Run the server and client using MQTT

If you want to run the edge server and client using MQTT, you need to run the following commands.

```bash
bash run_server.sh your_run_id

# in a new terminal window

# run the client 1
bash run_client.sh 1 your_run_id

# run the client with client_id
bash run_client.sh [CLIENT_ID] your_run_id
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

## How to Add Your Own Model?

Our framework supports `PyTorch` based models. To add your own specific model,

1. Create a `PyTorch` model and place it under `model` folder.
2. Prepare a `trainer module` by inheriting the base class `ClientTrainer`.
3. Prepare an experiment file similar to `fedml_*.py` and shell script similar to `run_*.sh`.
4. Adjust the `fedml_config.yaml` file with the model-specific parameters.

## How to Add More Datasets, Domain-Specific Splits & Non-I.I.D.ness Generation Mechanisms?

Create new folder for your dataset under `data/` folder and provide utilities to before feeding the data to federated pre-processing utilities listed in `data/data_loader.py` based on your new dataset.

Splits and Non-I.I.D.'ness methods specific to each task are also located under `data/data_loader.py`. By default, we provide I.I.D. and non-I.I.D. sampling, Dirichlet distribution sampling based on sample size of the dataset. To create custom splitting method based on the sample size, you can create a new function or modify `load_partition_data_*` function.

## Code Structure of healthcare

- `config`: Experiment and GPU mapping configurations.

- `data`: Provide data downloading scripts and store the downloaded datasets. FedCV supports more advanced datasets and models for federated training of computer vision tasks.
- `model`: advanced CV models.
- `trainer`: please define your own trainer.py by inheriting the base class in `fedml.core.alg_frame.client_trainer.ClientTrainer `. Some tasks can share the same trainer.
- `utils`: utility functions.

You can see the `README.md` file in each folder for more details.

## Citation

Please cite our FedML papers if it helps your research.

```text
@misc{he2020fedml,
      title={FedML: A Research Library and Benchmark for Federated Machine Learning},
      author={Chaoyang He and Songze Li and Jinhyun So and Xiao Zeng and Mi Zhang and Hongyi Wang and Xiaoyang Wang and Praneeth Vepakomma and Abhishek Singh and Hang Qiu and Xinghua Zhu and Jianzong Wang and Li Shen and Peilin Zhao and Yan Kang and Yang Liu and Ramesh Raskar and Qiang Yang and Murali Annavaram and Salman Avestimehr},
      year={2020},
      eprint={2007.13518},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contact

Please find contact information at the homepage.
