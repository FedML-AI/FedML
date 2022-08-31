## KiTS19

KiTS19 dataset is an open access Kidney Tumor Segmentation dataset that was made public in 2019 for a segmentation Challenge (https://kits19.grand-challenge.org/data/).
We use the official KiTS19 repository (https://github.com/neheller/kits19) to download the dataset.

## License:

Find attached the link to [the full license](https://data.donders.ru.nl/doc/dua/CC-BY-NC-SA-4.0.html?0) and [dataset terms](https://kits19.grand-challenge.org/data/).

## Dataset Description

|             | Dataset description                                                                                                                                                                                                                                                                       |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Description | This is the dataset from KiTS19 Challenge.                                                                                                                                                                                                                                                |
| Dataset     | 210 CT scans with segmentation masks as Train Data and 90 CT scans with no segmentations as Test Data. Since Test data does not have ground truth segmentation masks, we cannot use it for training/testing. Therefore, we will use only 210 CT scans in our exploration of this dataset. |
| Centers     | Data comes from 87 different centers. The sites information can be found in fed_kits19/dataset_creation_scripts/anony_sites.csv file. We include only those silos (total of 6) that have greater than 10 data samples (images), which leaves us with 96 patients data samples.            |
| Task        | Supervised Segmentation                                                                                                                                                                                                                                                                   |

## Data Download instructions

The commands for data download
(as given on the official kits19 git repository (https://github.com/neheller/kits19)) are as follows,

1. Cd to a different directory with sufficient space to hold kits data (~30GB) and clone the kits19 git repository:

```bash
git clone https://github.com/neheller/kits19 ~/healthcare/kits19
```

If you want to customize the data path, you also need to change the `data_cache_dir` variable in `config/fedml_config.yaml` and `config/simulation/fedml_config.yaml`.

2. Proceed to read and accept the license and data terms

3. Run the following commands to download the dataset. Make sure you have ~30GB space available.

```bash
cd ~/healthcare/kits19
pip install -r requirements.txt
python -m starter_code.get_imaging
```

These commands will populate the data folder (given in the kits19 repository) with the imaging data.

## Data Preprocessing

For preprocessing, we use [nnunet](https://github.com/MIC-DKFZ/nnUNet) library and [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators) packages. We exploit nnunet preprocessing pipeline
to apply intensity normalization, voxel and foreground resampling. In addition, we apply extensive transformations such as random crop, rotation, scaling, mirror etc from the batchgenerators package.

1. Environment setup

```bash
git clone https://github.com/FedML-AI/FedML

cd FedML/python/app/healthcare/fed_kits19
pip install -r requirements.txt

cd config/
bash bootstrap.sh

cd ..
pip install flamby[kits19]
```

2. Run the MPI simulation

```bash
bash run_simulation.sh [CLIENT_NUM]
```

There are only 6 center in this case, so the maximum number of clients is 6.

3. Run the edge server and client using MQTT or on MLOps

If you want to run the edge server and client using MQTT, you need to run the following commands.

```bash
bash run_server.sh your_run_id

# in a new terminal window
bash run_client.sh [CLIENT_ID] your_run_id
```

Client ID is the id of the client from 1 to 6.

4. Data preprocessing and debug mode

If you already have the data preprocessed, or you want to debug the preprocessing, you can change the `preprocessed` variable and `debug` variable in `config/fedml_config.yaml` or `config/simulation/fedml_config.yaml` to `true`.

```bash
data_args:
  dataset: "Fed-KITS2019"
  data_cache_dir: ~/healthcare/kits19
  partition_method: "hetero"
  partition_alpha: 0.5
  debug: false # change here! flamby: debug or not
  preprocessed: false # change here! flamby: preprocessed or not, need to preprocess in first

```

**Warning:** If you use more threads than your machine has available CPUs it, the preprocessing can halt indefinitely.
With this preprocessing, running the experiments can be very time efficient as it saves the preprocessing time for every experiment run.

Note that estimated memory requirement for this training is around 14.5 GB.

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
