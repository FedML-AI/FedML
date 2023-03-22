## Camelyon16

Camelyon16 as Camelyon17 are open access (CC0), the original dataset is accessible [here](https://camelyon17.grand-challenge.org/Data/).
We will use the [Google-Drive-API-v3](https://developers.google.com/drive/api/v3/quickstart/python) in order to fetch the slides from the public Google Drive and will then tile the matter using a feature extractor producing a bag of features for each slide.

## Dataset description

|                    | Dataset description
|--------------------| -----------------------------------------------------------------------------------------------
| Description        | Dataset from Camelyon16
| Dataset size       | 900 GB (and 50 GB after features extraction).
| Centers            | 2 centers - RUMC and UMCU.
| Records per center | RUMC: 169 (Train) + 74 (Test), UMCU: 101 (Train) + 55 (Test)
| Inputs shape       | Tensor of shape (10000, 2048) (after feature extraction).
| Total nb of points | 399 slides.
| Task               | Weakly Supervised (Binary) Classification.


## Download and preprocessing instructions

### Introduction
In order to use the Google Drive API you need to have a google account and to access the [google developpers console](https://console.cloud.google.com/apis/credentials/consent?authuser=1) in order to get a json containing an OAuth2.0 secret.  

All steps necessary to obtain the JSON are described in numerous places in the internet such as in pydrive's [quickstart](https://pythonhosted.org/PyDrive/quickstart.html), or in this [very nice tutorial's first 5 minutes](https://www.youtube.com/watch?v=1y0-IfRW114) on Youtube.
It should not take more than 5 minutes. The important steps are listed below.

### Step 1: Setting up Google App and associated secret

1. Create a project in [Google console](https://console.cloud.google.com/apis/credentials/consent?authuser=1). For instance, you can call it `flamby`.
2. Go to Oauth2 consent screen (on the left of the webpage), choose a name for your app and publish it for external use.
3. Go to Credentials, create an id, then client oauth id  
4. Choose Web app, go through the steps and **allow URI redirect** towards http://localhost:6006 and http://localhost:6006/ (notice the last backslash)
5. Retrieve the secrets in JSON by clicking on Download icon at the end of the process.
6. Enable Google Drive API for this project, by clicking on "API and services" on the left panel

Then copy-paste your secrets to the directory you want: 
~/healthcare/<secret-json>

## Data Download instructions

Set download in the config file to True. Note that please only set download to True at the first time you run the code.

```yaml
data_args:
  dataset: "Fed-Camelyon16"
  data_cache_dir: ~/healthcare/camelyon16
  secret_path: ~/healthcare/client_secret_1050214326187-s1vt68rfjc6daf09gab6163d871sti4r.apps.googleusercontent.com.json
  partition_method: "hetero"
  partition_alpha: 0.5
  debug: false # flamby: debug or not
  preprocessed: false # flamby: preprocessed or not, need to preprocess in first
  download: true # flamby: download or not
  tile_batch_size: 64
  num_workers_tile: 4
  tile_from_scratch: false
  remove_big_tiff: false
```

**IMPORTANT :** If you choose to relocate the dataset after downloading it, it is
imperative that you run the following script otherwise all subsequent scripts will not find it:

```
python update_config.py --new-path /new/path/towards/dataset
```

## Training

1. Environment setup

```bash
git clone https://github.com/FedML-AI/FedML

cd FedML/python/app/healthcare/fed_camelyon16
pip install -r requirements.txt

cd config/
bash bootstrap.sh

cd ..
pip install flamby[camelyon16]
```

2. Run the MPI simulation

```bash
bash run_simulation.sh [CLIENT_NUM]
```

There are only 2 center in this case, so the maximum number of clients is 2.

3. Run the edge server and client using MQTT or on MLOps

If you want to run the edge server and client using MQTT, you need to run the following commands.

```bash
bash run_server.sh your_run_id

# in a new terminal window
bash run_client.sh [CLIENT_ID] your_run_id
```

Client ID is the id of the client from 1 to 2.

# Citation:

```bash
@article{he2021fedcv,
  title={Fedcv: a federated learning framework for diverse computer vision tasks},
  author={He, Chaoyang and Shah, Alay Dilipbhai and Tang, Zhenheng and Sivashunmugam, Di Fan1Adarshan Naiynar and Bhogaraju, Keerti and Shimpi, Mita and Shen, Li and Chu, Xiaowen and Soltanolkotabi, Mahdi and Avestimehr, Salman},
  journal={arXiv preprint arXiv:2111.11066},
  year={2021}
}
@misc{he2020fedml,
      title={FedML: A Research Library and Benchmark for Federated Machine Learning},
      author={Chaoyang He and Songze Li and Jinhyun So and Xiao Zeng and Mi Zhang and Hongyi Wang and Xiaoyang Wang and Praneeth Vepakomma and Abhishek Singh and Hang Qiu and Xinghua Zhu and Jianzong Wang and Li Shen and Peilin Zhao and Yan Kang and Yang Liu and Ramesh Raskar and Qiang Yang and Murali Annavaram and Salman Avestimehr},
      year={2020},
      eprint={2007.13518},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
