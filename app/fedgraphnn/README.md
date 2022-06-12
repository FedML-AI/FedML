、# FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks
A Research-oriented Federated Learning Library and Benchmark Platform for Graph Neural Networks. 
Accepted to ICLR-DPML and MLSys21 - GNNSys'21 workshops. 


## Data Preparation

# FedGraphNN datasets

1. Graph - level 
      1. MoleculeNet -> We provide preprocessed versions of MoleculeNet datasets. To use datasets,first run  ```bash download_and_unzip.sh```  located under each dataset folder in  ```data/moleculenet```
      2. Social Networks -> We use PyTorch Geometric datasets for our social network datasets. For details, please see [this link](https://github.com/FedML-AI/FedGraphNN/blob/main/data_preprocessing/social_networks/data_loader.py)
2. Sub-graph Level
      1. Knowledge Graphs -> Please first run bash file inside ```data/subgraph-level```
      2. Recommendation Systems -> We provide preprocessed versions of Ciao and Epinions
3. Node-level
      1. Coaauthor & Citation Networks (Ego Networks) -> [Details](https://github.com/FedML-AI/FedGraphNN/tree/main/experiments/distributed/ego_networks)


## Experiments 

1. Graph Level
      1. MoleculeNet [Centralized Experiments](https://github.com/FedML-AI/FedGraphNN/tree/main/experiments/centralized) [Federated Experiments](https://github.com/FedML-AI/FedGraphNN/tree/main/experiments/distributed/moleculenet) 
      2.  Social Networks [Federated Experiments](https://github.com/FedML-AI/FedGraphNN/tree/main/experiments/distributed/social_networks)
2. Sub-graph Level
      1. Recommendation Systems [Federated Experiments](https://github.com/FedML-AI/FedGraphNN/tree/main/experiments/distributed/recommender_system)
3. Node Level
      1. Ego Networks (Citation & Coauthor Networks) [Federated Experiments](https://github.com/FedML-AI/FedGraphNN/tree/main/experiments/distributed/ego_networks)

## How to Add Your Own Model?
Our framework supports [PyTorch](https://github.com/FedML-AI/FedGraphNN/tree/main/model/moleculenet) and [PyTorch Geometric](https://github.com/FedML-AI/FedGraphNN/blob/main/model/recommender_system/sage_link.py) based models. To do so, 

1. Create a Pytorch/PyG based model and place it under model folder
2. Prepare a trainer module ([example](https://github.com/FedML-AI/FedGraphNN/blob/main/training/subgraph_level/fed_subgraph_lp_trainer.py)) by inheriting the base class in `FedML/fedml-core/trainer/fedavg_trainer.py`.
3. Prepare an experiment file similar to files in `experiments/` folder.

## How to Add More Datasets ? 
If it is a PyTorch Geometric dataset, please see [this link](https://github.com/FedML-AI/FedGraphNN/blob/main/data_preprocessing/social_networks/data_loader.py)

Otherwise, do the following:
1. Create new folder under `data_preprocessing` folder and re-define `data_preprocessing/data_loader.py` based on your new dataset.
2. Rewrite `data_loader.py` file under `data_preprocessing` folder


## How to Add Domain-Specific Splits & Non-I.I.D.ness Generation Mechanism?

Splits and Non-I.I.D.'ness methods are located under `data_preprocessing` library. By default, we provide I.I.D. and non-I.I.D. sampling(`create_non_uniform_split.py` , Dirichlet distribution sampling) based on sample size of the dataset.

To create custom splitting method based on the sample size, you can create a new function or modify `create_non_uniform_split.py` function.


## Code Structure of FedGraphNN
<!-- Note: The code of FedGraphNN only uses `FedML/fedml_core` and `FedML/fedml_api`.
In near future, once FedML is stable, we will release it as a python package. 
At that time, we can install FedML package with pip or conda, without the need to use Git submodule. -->

- `data`: Provide data downloading scripts and store the downloaded datasets.
Note that in `FedML/data`, there also exists datasets for research, but these datasets are used for evaluating federated optimizers (e.g., FedAvg) and platforms. FedGraphNN supports more advanced datasets and models for federated training of graph neural networks. 

- `data_preprocessing`: Domain-specific PyTorch/PyG Data Loaders for centralized and distributed training. 

- `model`: GNN models written in Pytorch/PyG.

- `trainer`: please define your own `trainer.py` by inheriting the base class in `FedML/fedml-core/trainer/fedavg_trainer.py`.
Some tasks can share the same trainer.

- `experiments/distributed`: 
1. `experiments` is the entry point for training. It contains experiments in different platforms. We start from `distributed`.
1. Every experiment integrates FOUR building blocks `FedML` (federated optimizers), `data_preprocessing`, `model`, `trainer`.
3. To develop new experiments, please refer the code at `experiments/distributed/text-classification`.

- `experiments/centralized`: 
1. please provide centralized training script in this directory. 
2. This is used to get the reference model accuracy for FL. 
3. You may need to accelerate your training through distributed training on multi-GPUs and multi-machines. Please refer the code at `experiments/centralized/DDP_demo`.


## Citation
Please cite our FedML and FedGraphNN papers if it helps your research.
You can describe us in your paper like this: "We develop our experiments based on FedML".
```
@misc{he2021fedgraphnn,
      title={FedGraphNN: A Federated Learning System and Benchmark for Graph Neural Networks}, 
      author={Chaoyang He and Keshav Balasubramanian and Emir Ceyani and Carl Yang and Han Xie and Lichao Sun and Lifang He and Liangwei Yang and Philip S. Yu and Yu Rong and Peilin Zhao and Junzhou Huang and Murali Annavaram and Salman Avestimehr},
      year={2021},
      eprint={2104.07145},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
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

 
