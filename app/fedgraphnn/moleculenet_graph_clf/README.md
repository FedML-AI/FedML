# MoleculeNet - Graph Property Classification


## Motivation

Molecular ML has been maturing rapidly over the last few years thanks to the graph neural networks. Due to privacy concerns and regulatoy restrictions, it is hard to collect users and/or organizations data into a single silo. This is especially crucial for bioinformatics and drug discovery applciations.  However, algorithmic progress has been limited due to the lack of a standard benchmark to compare the efficacy of proposed methods; most new algorithms are benchmarked on different datasets making it challenging to gauge the quality of proposed methods. Learnable representations still struggle to deal with complex tasks under data scarcity and highly imbalanced classification. 

## Dataset Preparation
Before training, each dataset has to be downloaded first from our servers. 

```
cd data/sider
sh download_and_unzip.sh
```


## Training
Before starting training, make sure that setup with  `config/fedml_config.yaml` is correct. 
```
WORKSPACE=./FedML/app/fedgraphnn/applications/moleculenet_graph_clf
cd $WORKSPACE

sh run_moleculenet_clf.sh 4
```