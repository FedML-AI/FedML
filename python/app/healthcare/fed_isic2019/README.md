# ISIC 2019

The dataset used in this repo comes from the [ISIC2019 challenge](https://challenge.isic-archive.com/landing/2019/) and the [HAM1000 database](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).
We do not own the copyright of the data, everyone using those datasets should abide by their licences (see below) and give proper attribution to the original authors.

## Dataset description

|             | Dataset description                                                                                  |
| ----------- | ---------------------------------------------------------------------------------------------------- |
| Description | Dataset from the ISIC 2019 challenge, we keep images for which the datacenter can be extracted.      |
| Dataset     | 23,247 images of skin lesions ((9930/2483), (3163/791), (2691/672), (1807/452), (655/164), (351/88)) |
| Centers     | 6 centers (BCN, HAM_vidir_molemax, HAM_vidir_modern, HAM_rosendahl, MSK, HAM_vienna_dias)            |
| Task        | Multiclass image classification                                                                      |

The "ISIC 2019: Training" is the aggregate of the following datasets:

BCN_20000 Dataset: (c) Department of Dermatology, Hospital Clínic de Barcelona

HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; [HAM10000 dataset](https://doi.org/10.1038/sdata.2018.161)

MSK Dataset: (c) Anonymous; [challenge 2017](https://arxiv.org/abs/1710.05006); [challenge 2018](https://arxiv.org/abs/1902.03368)

## Data

To download the ISIC 2019 training data and extract the original datacenter information for each image,
First cd into the `dataset_creation_scripts` folder:

```bash
cd flamby/datasets/fed_isic2019/dataset_creation_scripts
```

then run:

```
python download_isic.py --output-folder ~/healthcare/isic2019
```

The file train_test_split contains the train/test split of the images (stratified by center).

## Image preprocessing

To preprocess and resize images, run:

```
python resize_images.py
```

This script will resize all images so that the shorter edge of the resized image is 224px and the aspect ratio of the input image is maintained.
[Color constancy](https://en.wikipedia.org/wiki/Color_constancy) is added in the preprocessing.

**Be careful: in order to allow for augmentations, images aspect ratios are conserved in the preprocessing so images are rectangular with a fixed width so they all have different heights. As a result they cannot be batched without cropping them to a square. An example of such a cropping strategy can be found in the benchmark found below.**

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
bash run_server.sh your_run_id

# in a new terminal window
bash run_client.sh [CLIENT_ID] your_run_id
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
