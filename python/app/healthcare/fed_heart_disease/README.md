## Heart Disease

The Heart Disease dataset [1] was collected in 1988 in four centers:
Cleveland, Hungary, Switzerland and Long Beach V. We do not own the
copyright of the data: everyone using this dataset should abide by its
licence and give proper attribution to the original authors. It is
available for download
[here](https://archive-beta.ics.uci.edu/ml/datasets/heart+disease).

## Dataset description

|                    | Dataset description                                           |
| ------------------ | ------------------------------------------------------------- |
| Description        | Heart Disease dataset.                                        |
| Dataset size       | 39,6 KB.                                                      |
| Centers            | 4 centers - Cleveland, Hungary, Switzerland and Long Beach V. |
| Records per center | Train/Test: 199/104, 172/89, 30/16, 85/45.                    |
| Inputs shape       | 16 features (tabular data).                                   |
| Total nb of points | 740.                                                          |
| Task               | Binary classification                                         |

## Data Download instructions

Set download in the config file to True. Note that please only set download to True at the first time you run the code.

```yaml
data_args:
  dataset: "Fed-Heart-Disease"
  data_cache_dir: ~/healthcare/heart_disease # flamby
  partition_method: "hetero"
  partition_alpha: 0.5
  debug: false # flamby: debug or not
  preprocessed: false # flamby: preprocessed or not, need to preprocess in first
  download: true # flamby: download or not
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

cd FedML/python/app/healthcare/fed_heart_disease
pip install -r requirements.txt

cd config/
bash bootstrap.sh

cd ..
pip install flamby[heart_disease]
```

2. Run the MPI simulation

```bash
bash run_simulation.sh [CLIENT_NUM]
```

There are only 4 center in this case, so the maximum number of clients is 4.

3. Run the edge server and client using MQTT or on MLOps

If you want to run the edge server and client using MQTT, you need to run the following commands.

```bash
bash run_server.sh

# in a new terminal window
bash run_client.sh [CLIENT_ID]
```

Client ID is the id of the client from 1 to 4.

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
