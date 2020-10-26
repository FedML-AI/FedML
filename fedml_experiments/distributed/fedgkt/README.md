# Group Knowledge Transfer: Federated Learning of Large CNNs at the edge

This repository is the official implementation of [Group Knowledge Transfer: Federated Learning of Large CNNs on the edge](https://arxiv.org/abs/2007.14513). 


## 1. Introduction

<img src="https://i2.wp.com/chaoyanghe.com/wp-content/uploads/2020/10/FedGKT_framework.png" alt="FedGKT"/>
Scaling up the convolutional neural network (CNN) size (e.g., width, depth, etc.) is known to effectively improve model accuracy. However, the large model size impedes training on resource-constrained edge devices. For instance, federated learning (FL) may place undue burden on the compute capability of edge nodes, even though there is a strong practical need for FL due to its privacy and confidentiality properties. To address the resource-constrained reality of edge devices, we reformulate FL as a group knowledge transfer training algorithm, called FedGKT. FedGKT designs a variant of the alternating minimization approach to train small CNNs on edge nodes and periodically transfer their knowledge by knowledge distillation to a large server-side CNN. FedGKT consolidates several advantages into a single framework: reduced demand for edge computation, lower communication bandwidth for large CNNs, and asynchronous training, all while maintaining model accuracy comparable to FedAvg. We train CNNs designed based on ResNet-56 and ResNet-110 using three distinct datasets (CIFAR-10, CIFAR-100, and CINIC-10) and their non-I.I.D. variants. Our results show that FedGKT can obtain comparable or even slightly higher accuracy than FedAvg. More importantly, FedGKT makes edge training affordable. Compared to the edge training using FedAvg, FedGKT demands 9 to 17 times less computational power (FLOPs) on edge devices and requires 54 to 105 times fewer parameters in the edge CNN. Our source code is released at https://fedml.ai.


\
Check the Video here:
https://studio.slideslive.com/web_recorder/share/20201022T054207Z__NeurIPS_posters__18310__group-knowledge-transfer-coll?s=084b45a3-0768-4590-abfe-744506af2a3c

## 2. Installation
http://doc.fedml.ai/#/installation-distributed-computing

## 3. Experiments

## ResNet56 GKT
#### CIFAR10
```
sh run_FedGKT.sh 0 cifar10 homo 200 1 20 Adam 0.001 1 0 resnet56 fedml_resnet56_homo_cifar10 "./../../../data/cifar10" 256

##run on background
nohup sh run_FedGKT.sh 0 cifar10 homo 200 1 20 Adam 0.001 1 0 resnet56 fedml_resnet56_homo_cifar10 "./../../../data/cifar10" 256 > ./FedGKT_resnet56_homo_cifar10.log 2>&1 &

```

#### CIFAR100 
```
sh run_FedGKT.sh 0 cifar100 homo 200 1 20 Adam 0.001 1 0 resnet56 fedml_resnet56_homo_cifar100 "./../../../data/cifar100" 256

##run on background
nohup sh run_FedGKT.sh 0 cifar100 homo 200 1 20 Adam 0.001 1 0 resnet56 fedml_resnet56_homo_cifar100 "./../../../data/cifar100" 256 > ./FedGKT_resnet56_homo_cifar100.log 2>&1 &
```

#### CINIC10
```
sh run_FedGKT.sh 0 cinic10 homo 200 1 20 Adam 0.001 1 0 resnet56 fedml_resnet56_homo_cinic10 "./../../../data/cinic10" 256

##run on background
nohup sh run_FedGKT.sh 0 cinic10 homo 200 1 20 Adam 0.001 1 0 resnet56 fedml_resnet56_homo_cinic10 "./../../../data/cinic10" 256 > ./FedGKT_resnet56_homo_cinic10.log 2>&1 &
```

## 4. Results
Please read the experiment section in our paper.

## 5. Citation
If you use any part of this code in your research or any engineering project, please cite our paper: 
```
@inproceedings{FedGKT2020,
    title={Group Knowledge Transfer: Federated Learning of Large CNNs at the Edge},
    author={He, Chaoyang and Annavaram, Murali and Avestimehr, Salman},
    booktitle = {Advances in Neural Information Processing Systems 33},
    year={2020}
}
```

```
@article{chaoyanghe2020fedml,
  Author = {He, Chaoyang and Li, Songze and So, Jinhyun and Zhang, Mi and Wang, Hongyi and Wang, Xiaoyang and Vepakomma, Praneeth and Singh, Abhishek and Qiu, Hang and Shen, Li and Zhao, Peilin and Kang, Yan and Liu, Yang and Raskar, Ramesh and Yang, Qiang and Annavaram, Murali and Avestimehr, Salman},
  Journal = {arXiv preprint arXiv:2007.13518},
  Title = {FedML: A Research Library and Benchmark for Federated Machine Learning},
  Year = {2020}
}
```

## 6. Contributing
If you would like to contribute any improvement to this source code, please contact the authors by Emails.

## 7. Contacts

Chaoyang He \
https://chaoyanghe.com \
chaoyang.he@usc.edu \
chaoyanghe.com@gmail.com
