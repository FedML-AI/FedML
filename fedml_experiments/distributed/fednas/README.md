# FedNAS: Federated Deep Learning via Neural Architecture Search
This is the source code for the following paper:
> [FedNAS: Federated Deep Learning via Neural Architecture Search](https://arxiv.org/abs/2004.08546) \
> Chaoyang He, Murali Annavaram, Salman Avestimehr

## 1. Installation
http://doc.fedml.ai/#/installation-distributed-computing

## 2. Hardware Requirements
We set up our experiment in a distributed computing network equipped with GPUs. 
There are 17 nodes in total, one representing the server-side, and the other 16 nodes representing clients, which can be organizations in the real world (e.g., the hospitals). 
Each node is a physical server that has an NVIDIA RTX 2080Ti GPU card inside. 
we assume all the clients join the training process for every communication round.


## 3. Experiments
Once the hardware and software environment are both ready, you can easily use the following command to run FedNAS.
Note:
1. you may find other packages are missing. Please install accordingly by "conda" or "pip".
2. Our default setting is 16 works. Please change parameters in "run_fed_nas_search.sh" based on your own physical servers and requirements.
3. Please adjust the batch size or worker number to fit your own physical machine configuration.

- Homogeneous distribution (IID) experiment:
```
# search
sh run_fednas_search.sh 1 4 darts homo 50 5 64

# train
sh run_fednas_train.sh 1 4 darts homo 500 15 64
```

- Heterogeneous distribution (Non-IID) experiment:
```
# search
sh run_fednas_search.sh 1 4 darts hetero 50 5 64

# train
sh run_fednas_train.sh 1 4 darts hetero 500 15 64
```

We can also run code in a single 4 x NVIDIA RTX 2080Ti GPU server. 
In this case, we should decrease the batch size to 2 to guarantee that the total 17 processes can be loaded into the memory. 
The running script for such setting is:
```
# search
sh run_fednas_search.sh 4 darts hetero 50 5 8
```


## 4. Citations
If you use any part of this code in your research or any engineering project, please cite our paper: 

```
@inproceedings{FedNAS,
  title={FedNAS: Federated Deep Learning via Neural Architecture Search},
  author={He, Chaoyang and Annavaram, Murali and Avestimehr, Salman},
  booktitle={CVPR 2020 Workshop on Neural Architecture Search and Beyond for Representation Learning},
  year={2020},
}
```

```
@inproceedings{MiLeNAS,
  title={MiLeNAS: Efficient Neural Architecture Search via Mixed-Level Reformulation},
  author={He, Chaoyang and Ye, Haishan and Shen, Li and Zhang, Tong},
  booktitle={Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020},
}
```
