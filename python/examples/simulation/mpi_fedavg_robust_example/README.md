## Installation
http://doc.fedml.ai

## Supported Attack and Defense Methods:
### Attacks:
1. The "Edge-case" Black-box Attack (NeurIPS 2020) [Link](https://arxiv.org/abs/2007.05084)
2. (TODO) The "Edge-case" PGD Attack with Model Replacement (NeurIPS 2020) [Link](https://arxiv.org/abs/2007.05084)
3. (TODO) The "Edge-case" PGD Attack without Model Replacement (NeurIPS 2020) [Link](https://arxiv.org/abs/2007.05084)

### Defenses:
1. The Norm Difference Clipping Defense [Link](https://arxiv.org/abs/1911.07963)
2. The Weakly Differentialy Private Defense (Weak-DP) [Link](https://arxiv.org/abs/1911.07963)
3. (TODO) The RFA Defense [Link](https://arxiv.org/abs/1912.13445)
4. (TODO) The Krum/Multi-Krum Defenses (NeurIPS 2017) [Link](https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent)

## Run Experiments

#### CIFAR10 + Southwest Edge Case Backdoor (Black-box) under Weak-DP Defense
```
sh run_fedavg_robust.sh 4
```

