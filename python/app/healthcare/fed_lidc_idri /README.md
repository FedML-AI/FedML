# LIDC-IDRI

This dataset comes from the [TCIA GDC data portal](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI#1966254a2b592e6fba14f949f6e23bb1b7804cc).
We do not own the copyright of the data, everyone using this dataset should abide by [its licence](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI#1966254a2b592e6fba14f949f6e23bb1b7804cc) and give proper attribution to the original authors.
We therefore give credit here to:
### Data Citation

Armato III, S. G., McLennan, G., Bidaut, L., McNitt-Gray, M. F., Meyer, C. R., Reeves, A. P., Zhao, B., Aberle, D. R., Henschke, C. I., Hoffman, E. A., Kazerooni, E. A., MacMahon, H., Van Beek, E. J. R., Yankelevitz, D., Biancardi, A. M., Bland, P. H., Brown, M. S., Engelmann, R. M., Laderach, G. E., Max, D., Pais, R. C. , Qing, D. P. Y. , Roberts, R. Y., Smith, A. R., Starkey, A., Batra, P., Caligiuri, P., Farooqi, A., Gladish, G. W., Jude, C. M., Munden, R. F., Petkovska, I., Quint, L. E., Schwartz, L. H., Sundaram, B., Dodd, L. E., Fenimore, C., Gur, D., Petrick, N., Freymann, J., Kirby, J., Hughes, B., Casteele, A. V., Gupte, S., Sallam, M., Heath, M. D., Kuhn, M. H., Dharaiya, E., Burns, R., Fryd, D. S., Salganicoff, M., Anand, V., Shreter, U., Vastagh, S., Croft, B. Y., Clarke, L. P. (2015). Data From LIDC-IDRI [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX

### Publication Citation

Armato SG 3rd, McLennan G, Bidaut L, McNitt-Gray MF, Meyer CR, Reeves AP, Zhao B, Aberle DR, Henschke CI, Hoffman EA, Kazerooni EA, MacMahon H, Van Beeke EJ, Yankelevitz D, Biancardi AM, Bland PH, Brown MS, Engelmann RM, Laderach GE, Max D, Pais RC, Qing DP, Roberts RY, Smith AR, Starkey A, Batrah P, Caligiuri P, Farooqi A, Gladish GW, Jude CM, Munden RF, Petkovska I, Quint LE, Schwartz LH, Sundaram B, Dodd LE, Fenimore C, Gur D, Petrick N, Freymann J, Kirby J, Hughes B, Casteele AV, Gupte S, Sallamm M, Heath MD, Kuhn MH, Dharaiya E, Burns R, Fryd DS, Salganicoff M, Anand V, Shreter U, Vastagh S, Croft BY.  The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. Medical Physics, 38: 915--931, 2011. DOI: https://doi.org/10.1118/1.3528204

### TCIA Citation

Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P., Moore, S., Phillips, S., Maffitt, D., Pringle, M., Tarbox, L., & Prior, F. (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. Journal of Digital Imaging, 26(6), 1045–1057. https://doi.org/10.1007/s10278-013-9622-7

Additionally here we preprocess the dataset to create nifti files and retrieve the per-center metadata.

## Dataset description

|                   | Dataset description |
| ----------------- | -----------------------------------------------|
| Description       | This is the dataset from LIDC-IDRI study |
| Dataset           | 1018 CT-scans with masks (GE MEDICAL SYSTEMS 670 (536 / 134), SIEMENS 205 (164, 41), TOSHIBA 69 (55 / 14), Philips 74 (59 / 15) |
| Centers           | Manufacturer of the CT-scans (GE MEDICAL SYSTEMS, SIEMENS, TOSHIBA, Philips) |
| Permission        | Public |
| Task              | Segmentation |
| Format            | CT-scans and masks in the `nifti` format |

This dataset was used in the [Luna16 challenge](https://luna16.grand-challenge.org/Home/).
Note that contrary to the challenge, in which slides with thickness larger than 3 mm were removed, 
the present dataset contains all 1018 slides from [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI).

The data is split in a training and testing sets containing 80% and 20% of the available scans respectively.
This split is stratified according to the centers (the manufacturers of the CT-scan apparatus),
in order to preserve the distribution of centers of origin within each split. 

Centers (i.e., manufacturers) are encoded using a unique index, as follows: GE MEDICAL SYSTEMS: 0, Philips: 1, SIEMENS: 2, TOSHIBA: 3.

## Data Download instructions

Set download in the config file to True. Note that please only set download to True at the first time you run the code.

```yaml
data_args:
  dataset: "Fed-LIDC-IDRI"
  data_cache_dir: ~/healthcare/lidc_idri # flamby
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

cd FedML/python/app/healthcare/fed_lidc_idri
pip install -r requirements.txt

cd config/
bash bootstrap.sh

cd ..
pip install flamby[lidc_idri]
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
