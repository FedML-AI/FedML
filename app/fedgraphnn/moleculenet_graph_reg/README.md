# MoleculeNet - Graph Property Prediction

## Motivation

Molecular machine learning has been maturing rapidly over the last few years. Improved methods and the presence of larger datasets have enabled machine learning algorithms to make increasingly accurate predictions about molecular properties. However, algorithmic progress has been limited due to the lack of a standard benchmark to compare the efficacy of proposed methods; most new algorithms are benchmarked on different datasets making it challenging to gauge the quality of proposed methods. This work introduces MoleculeNet, a large scale benchmark for molecular machine learning. Learnable representations still struggle to deal with complex tasks under data scarcity and highly imbalanced classification.
## Data Preparation
Before training, each dataset has to be downloaded first. 
```
cd data/esol
sh download_and_unzip.sh
```

## Training
Before starting training, make sure that setup with  `config/fedml_config.yaml` is correct. Then, run the following script 
```
sh run_moleculenet_reg.sh 4
```