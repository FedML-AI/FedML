# Install FedML and Prepare the Distributed Environment
```
pip install fedml
```

# Introduction
https://github.com/chaoyanghe/FedNAS


## FedNAS: Federated Deep Learning via Neural Architecture Search
This is the source code for the following paper:
> [FedNAS: Federated Deep Learning via Neural Architecture Search](https://chaoyanghe.com/publications/FedNAS-CVPR2020-NAS.pdf)\
> Chaoyang He, Murali Annavaram, Salman Avestimehr \
> Accepted to <a href="https://sites.google.com/view/cvpr20-nas/" target="_blank">CVPR 2020 Workshop on Neural Architecture Search and Beyond for Representation Learning</a>

- Homogeneous distribution (IID) experiment:
```
# search
sh run_fednas_search.sh 4

# train
sh run_fednas_train.sh 4
```

- Heterogeneous distribution (Non-IID) experiment:
```
# search
sh run_fednas_search.sh 4

# train
sh run_fednas_train.sh 4
```

We can also run code in a single 4 x NVIDIA RTX 2080Ti GPU server. 
In this case, we should decrease the batch size to 2 to guarantee that the total 17 processes can be loaded into the memory. 
The running script for such setting is:
```
# search
sh run_fednas_search.sh 4 darts hetero 50 5 8
```


### 4. Citations
If you use any part of this code in your research or any engineering project, please cite our paper: 

```
@inproceedings{FedNAS,
  title={FedNAS: Federated Deep Learning via Neural Architecture Search},
  author={He, Chaoyang and Annavaram, Murali and Avestimehr, Salman},
  booktitle={CVPR 2020 Workshop on Neural Architecture Search and Beyond for Representation Learning},
  year={2020},
}
```

https://chaoyanghe.com/publications/FedNAS-CVPR2020-NAS.pdf

```
@inproceedings{MiLeNAS,
  title={MiLeNAS: Efficient Neural Architecture Search via Mixed-Level Reformulation},
  author={He, Chaoyang and Ye, Haishan and Shen, Li and Zhang, Tong},
  booktitle={Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020},
}
```




### 5. Contacts
Please feel free to contact me if you meet any problem when using this source code.
I am glad to upgrade the code meet to your requirements if it is reasonable.

I am also open to collaboration based on this elementary system and research idea.

> Chaoyang He \
> http://chaoyanghe.com