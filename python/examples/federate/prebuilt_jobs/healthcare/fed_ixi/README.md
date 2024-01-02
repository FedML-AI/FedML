# Fed IXI
## Data Citation

The IXI dataset is made available under the Creative Commons [CC BY-SA 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/legalcode). If you use the IXI data please acknowledge the source of the IXI data, e.g. the following website: https://brain-development.org/ixi-dataset/

IXI Tiny is derived from the same source. Acknowledge the following reference on TorchIO : https://torchio.readthedocs.io/datasets.html#ixitiny

### Publication Citation

Pérez-García F, Sparks R, Ourselin S. TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning. arXiv:2003.04696 [cs, eess, stat]. 2020. https://doi.org/10.48550/arXiv.2003.04696

## Introduction

This repository highlights **IXI** (*Information eXtraction from Images*), a medical dataset focusing on brain images through Structural magnetic resonance imaging (MRI), a non-invasive technique for examining the anatomy and pathology of the brain.

In the same register, we highlight a particular dataset called **IXI Tiny**, which is composed of preprocessed images from the standard **IXI** dataset. The idea behind the use of this dataset is to take advantage of its lightness, as well as the labels it directly provides so it allows us to handle an interesting segmentation task.

We have chosen to give some insight about the standard dataset, although we will focus on the lighter one, **IXI Tiny**, as a starting point. Note that the requirements for the standard IXI dataloader have been implemented, but for a matter of clarity, we will limit ourselves to the code of IXI tiny on this part of the repository. Of course, standard IXI could be added in the near future on a separate place.

## Standard IXI Dataset

### Overview

The **IXI** dataset contains “nearly 600 MR images from normal, healthy subjects”, including “T1, T2 and PD-weighted images, MRA images and Diffusion-weighted images (15 directions)”.

The dataset contains data from three different hospitals in London :
- Hammersmith Hospital using a Philips 3T system ([details of scanner parameters](http://wp.doc.ic.ac.uk/brain-development/scanner-philips-medical-systems-intera-3t/)).
- Guy’s Hospital using a Philips 1.5T system ([details of scanner parameters](http://wp.doc.ic.ac.uk/brain-development/scanner-philips-medical-systems-gyroscan-intera-1-5t/)).
- Institute of Psychiatry using a GE 1.5T system (details of the scan parameters not available at the moment).

For information, here is the respective size of the different archives:

| Modality | Size |
| :------: | ------ |
| T1 | 4.51G |
| T2 | 3.59G |
| PD | 3.79G |
| MRA | 11.5G |
| DTI | 3.98G |

**Total size**: 27.37G

Datapoints inside the different archives (our 5 modalities) follow this naming convention:

**IXI**[*patient id*]**-**[*hospital name*]**-**[*id*]**-**[*modality*]**.nii.gz**

These files contain images in **NIFTI** format.

## IXI Tiny Dataset

### Dataset description

|                    | Dataset description
|--------------------| -----------------------------------------------------------------------------------------------
| Description        | Dataset contains data from three different hospitals in London focusing on brain images through MRI.
| Dataset size       | 444 MB
| Centers            | 3 centers - Guys (Guy’s Hospital), HH (Hammersmith Hospital), IOP (Institute of Psychiatry).
| Records per center | Guys: 249/62, HH: 145/36, IOP: 59/15 (train/test).
| Inputs shape       | Image of shape (1, 48, 60, 48).
| Targets shape      | Image of shape (2, 48, 60, 48).
| Total nb of points | 566.
| Task               | Segmentation.

### Overview

**IXI Tiny** relies on **IXI**, a publicly available dataset of almost 600 subjects. This lighter version made by [TorchIO](https://torchio.readthedocs.io/datasets.html#ixitiny) is focusing on 566 T1-weighted brain MR images and comes with a set of corresponding labels (brain segmentations).

To produce the labels, ROBEX, an automatic whole-brain extraction tool for T1-weighted MRI data has been used.
Affine registration, which is a necessary prerequisite for many image processing tasks, has been performed using [NiftyReg](https://github.com/KCL-BMEIS/niftyreg) putting all the brain images onto a common reference space (MNI template). An orientation tweak has finally been made with [ITK](https://itk.org/).

Volumes have a dimension of 83 x 44 x 55 voxels (compared to 256 x 256 x 140 in the standard dataset).

The total size of this tiny dataset is 444 MB.

The structure of the archive containing the dataset has been modified, making it adapted for particular cases of use where each subject is represented by a directory containing all the modalities associated with them.

E.g.
```
IXI_sample   
│
└───IXI002-Guys-0828
│   │  
│   └───label
│   │   │   IXI002-Guys-0828_label.nii.gz
│   │  
│   └───T1
│   │   │   IXI002-Guys-0828_image.nii.gz
│   │  
│   └───T2
│   │   │   IXI002-Guys-0828_image.nii.gz
│   │ 
│   └───... 
│
└───IXI012-HH-1211
│   │  
│   └───label
│   │   │   IXI012-HH-1211_label.nii.gz
│   │  
│   └───T1
│   │   │   IXI012-HH-1211_image.nii.gz
│   │  
│   └───T2
│   │   │   IXI012-HH-1211_image.nii.gz
│   │ 
│   └───... 
│
│
└───...

```

## Data Download instructions

Set download in the config file to True. Note that please only set download to True at the first time you run the code.

```yaml
data_args:
  dataset: "Fed-IXI"
  data_cache_dir: ~/healthcare/ixi # flamby
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

cd FedML/python/app/healthcare/fed_ixi
pip install -r requirements.txt

cd config/
bash bootstrap.sh

cd ..
pip install flamby[lidc_ixi]
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

Client ID is the id of the client from 1 to 3.

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
