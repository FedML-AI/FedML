# MoleculeNet - Graph Property Classification

## Motivation

Molecular ML has been maturing rapidly over the last few years thanks to the graph neural networks. Due to privacy concerns and regulatory restrictions, it is hard to collect users and/or organizations data into a single silo. This is especially crucial for bioinformatics and drug discovery applications. Coupled with the aforementioned reasons, learnable representations still struggle to deal with complex tasks under data scarcity, heterogeneity, and highly imbalanced classification.  As an example, biomedical institutions might hold their own set of molecules which cannot be directly shared across silos due to commercial and privacy reasons. To simulate such scenarios, we utilize datasets from MoleculeNet.

## Dataset Preparation
Before training, each dataset has to be downloaded first from our servers.  For graph property classification task, we provide 9 datasets. To process `sider` dataset, all you have to do is to run the bash script under each dataset folder: 

```
cd data/sider
sh download_and_unzip.sh
```

Dataset preparation is same for all datasets.

## Training
Before starting training, make sure that setup with  `config/fedml_config.yaml` is correct. Then start the training with the desired number of GPU workers.
```
WORKSPACE=./FedML/app/fedgraphnn/applications/moleculenet_graph_clf
cd $WORKSPACE

sh run_distributed_simulation.sh 4
```