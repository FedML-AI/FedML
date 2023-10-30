# TCGA-BRCA
The dataset used in this repo comes from [The Cancer Genome Atlas (TCGA)](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga) from [the GDC data portal](https://portal.gdc.cancer.gov/). 
The data terms can be found [here](https://gdc.cancer.gov/access-data/data-access-processes-and-tools).
We do not guarantee that the use of this data can be done freely by the user. As such it is mandatory that one should check the applicability of the licence associated with this data before using it.


We selected one single cancer type: Breast Invasive Carcinoma (BRCA) and only use clinical tabular data. We replicate the preprocessing used by [Andreux et al.](https://arxiv.org/pdf/2006.08997.pdf) from data originally computed from TCGA by [Liu et al.](https://pubmed.ncbi.nlm.nih.gov/29625055/):

Liu J, Lichtenberg T, Hoadley KA, Poisson LM, Lazar AJ, Cherniack AD, Kovatich AJ, Benz CC, Levine DA, Lee AV, Omberg L, Wolf DM, Shriver CD, Thorsson V; Cancer Genome Atlas Research Network, Hu H. An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics. Cell. 2018 Apr 5;173(2):400-416.e11. doi: 10.1016/j.cell.2018.02.052. PMID: 29625055; PMCID: PMC6066282.

Andreux, M., Manoel, A., Menuet, R., Saillard, C., and Simpson, C., “Federated Survival Analysis with Discrete-Time Cox Models”, <i>arXiv e-prints</i>, 2020.

i.e. a subset of the features in the raw TCGA-BRCA dataset (categorical variables are one-hot encoded).


## Dataset description

|                    | Dataset description
|--------------------| -----------------------------------------------------------------------------------------------
| Description        | Clinical data from the TCGA-BRCA study with 1,088 patients.
| Dataset size       | 117,5 KB (stored in this repository).
| Centers            | 6 regions - Northeast, South, West, Midwest, Europe, Canada.
| Records per center | Train/Test: 248/63, 156/40, 164/42, 129/33, 129/33, 40/11.
| Inputs shape       | 39 features (tabular data).
| Targets shape      | (E,T). E: relative risk, continuous variable. T: ground truth, at 0 or 1. 
| Total nb of points | 1088.
| Task               | Survival analysis.


Raw TCGA-BRCA data can be viewed and downloaded [here](https://portal.gdc.cancer.gov/projects/TCGA-BRCA).

## Data
Preprocessed data is stored in this repo in the file ```brca.csv```, so the dataset does not need to be downloaded. The medical centers (with their geographic regions) are stored in the file ```centers.csv```. From this file and the patients' TCGA barcodes, we can extract the region of origin of each patient's tissue sort site (TSS). The numbers of sites being too large (64) we regroup them in 6 different regions (Northeast, South, West, Midwest, Europe, Canada). The patients' stratified split by region is static and stored in the train_test_split.csv file.

## Data Download instructions

Set download in the config file to True. Note that please only set download to True at the first time you run the code.

```yaml
data_args:
  dataset: "Fed-TCGA-BRCA"
  data_cache_dir: ~/healthcare/tcga_brca # flamby
  partition_method: "hetero"
  partition_alpha: 0.5
  debug: false # flamby: debug or not
  preprocessed: false # flamby: preprocessed or not, need to preprocess in first
  download: false # flamby: download or not
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

cd FedML/python/app/healthcare/fed_tcga_brca
pip install -r requirements.txt

cd config/
bash bootstrap.sh

cd ..
pip install flamby[tcga_brca]
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
