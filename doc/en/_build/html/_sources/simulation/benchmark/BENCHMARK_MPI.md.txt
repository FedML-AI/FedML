# Benchmarking Results for MPI-based federated learning
Please visit the following link to check the latest benchmark experimental results: [https://app.wandb.ai/automl/fedml/reports/FedML-Benchmark-Experimental-Results--VmlldzoxODE2NTU](https://app.wandb.ai/automl/fedml/reports/FedML-Benchmark-Experimental-Results--VmlldzoxODE2NTU) 
FedML white paper ([https://arxiv.org/pdf/2007.13518.pdf](https://arxiv.org/pdf/2007.13518.pdf)) also summarizes the dataset list and related benchmarks. 
We refer the hyper-parameters and reproduce results from many top-tier ML conferences. Please check details of our reference hyperparameters as follows.

### Linear Models
|     Data     |              Model              | Alg |   Partition  | #C | #C_p | bs | c_opt |      lr      | e | #R | acc |
|:----------------:|:-------------------------------:|:---------:|:-------------------:|:-------------------:|:--------------------:|:----------:|:------------:|:--------------:|:------:|:----------:|:--------:|
|       MNIST      |       LR       |   FedAvg  |      Power Law      |         1000        |          10          |     10     |        SGD       |     0.03     |    1   |    >100    |    >75   |
| Federated EMNIST |        LR       |   FedAvg |      Power Law      |         200         |          10          |     10     |        SGD       |     0.003    |    1   |    >200    |   10~40  |
|  Synthetic(α,β） |       LR       |   FedAvg  |      Power Law      |          30         |          10          |     10     |        SGD       |     0.01     |    1   |    >200    |    >60   |

Note: #C stands for client_num_in_total; #C_p stands for client_num_per_round; bs = batch_size; c_opt = client optimizer; e = epoch; #R = number of rounds; acc = accuracy. For Synthetic(α,β), (α,β) is chosen from (0,0), (0.5,0.5), (1,1)


- MNIST – Logistic Regression – FedAvg
    - Patition Method: ‘Federated optimization in heterogeneous networks’, page 7, Section 5.1, ‘Real data’
    - client_num_in_total: ‘Federated optimization in heterogeneous networks’, page 7, Section 5.1, ‘Real data’
    - client_num_per_round: ‘Federated optimization in heterogeneous networks’, page 18, Appendix C.2, ‘Hyperparameters’
    - batch_size: ‘Federated optimization in heterogeneous networks’, page 18, Appendix C.2, ‘Hyperparameters’
    - client_optimizer: ‘Federated optimization in heterogeneous networks’, page 8, Section 5.1, ‘Implementation
    - lr: ‘Federated optimization in heterogeneous networks’, page 18, Appendix C.2, ‘Hyperparameters’
    - epochs: ‘Federated optimization in heterogeneous networks’, page 21, Appendix C.3.2 Figure 9 description
    - comm_round: ‘Federated optimization in heterogeneous networks’, page 21, Appendix C.3.2 Figure 10
    - accuracy: ‘Federated optimization in heterogeneous networks’, page 21, Appendix C.3.2 Figure 10
- Federated EMNIST – Logistic Regression-FedAvg
    - Patition Method: ‘Federated optimization in heterogeneous networks’, page 7, Section 5.1, ‘Real data’
    - client_num_in_total: ‘Federated optimization in heterogeneous networks’, page 7, Section 5.1, ‘Real data’
    - client_num_per_round: ‘Federated optimization in heterogeneous networks’, page 18, Appendix C.2, ‘Hyperparameters’
    - batch_size: ‘Federated optimization in heterogeneous networks’, page 18, Appendix C.2, ‘Hyperparameters’
    - client_optimizer: ‘Federated optimization in heterogeneous networks’, page 8, Section 5.1, ‘Implementation
    - lr: ‘Federated optimization in heterogeneous networks’, page 18, Appendix C.2, ‘Hyperparameters’
    - epochs: ‘Federated optimization in heterogeneous networks’, page 21, Appendix C.3.2 Figure 9 description
    - comm_round: ‘Federated optimization in heterogeneous networks’, page 21, Appendix C.3.2 Figure 10
    - accuracy: ‘Federated optimization in heterogeneous networks’, page 21, Appendix C.3.2 Figure 10
- Synthetic(α,β) – Logistic Regression -FedAvg
    - Patition Method: ‘Federated optimization in heterogeneous networks’, page 18, Appendix C.1, ‘Synthetic’
    - client_num_in_total: ‘Federated optimization in heterogeneous networks’, page 18, Appendix C.1, ‘Synthetic’
    - client_num_per_round: ‘Federated optimization in heterogeneous networks’, page 18, Appendix C.2, ‘Hyperparameters’
    - batch_size: ‘Federated optimization in heterogeneous networks’, page 18, Appendix C.2, ‘Hyperparameters’
    - client_optimizer: ‘Federated optimization in heterogeneous networks’, page 8, Section 5.1, ‘Implementation
    - lr: ‘Federated optimization in heterogeneous networks’, page 18, Appendix C.2, ‘Hyperparameters’
    - epochs: ‘Federated optimization in heterogeneous networks’, page 8, Section 5.1, ‘Hyperparameters & evaluation metrics’
    - comm_round: ‘Federated optimization in heterogeneous networks’, page 19, Appendix C.3.2 Figure 6
    - accuracy: ‘Federated optimization in heterogeneous networks’, page 19, Appendix C.3.2 Figure 6
    
### Lightweight and shallow neural network models
|     Task     |     Data Set     |              Model              | Alogrithm |   Partition Method  | Partition Alpha | client_num_in_total | client_num_per_round | batch_size | client_optimizer |      lr      |       wd       | epochs | comm_round | accuracy |
|:----------------:|:----------------:|:-------------------------------:|:---------:|:-------------------:|:---------------:|:-------------------:|:--------------------:|:----------:|:----------------:|:------------:|:--------------:|:------:|:----------:|:--------:|
|       CV         | Federated EMNIST |       CNN (2 Conv + 2 FC)       |   FedAvg  |    Power Law        |        　       |         3400        |          10          |     20     |        SGD       |      0.1     |  -              |   1   |    >1500   |   84.9   |
|       CV         |     CIFAR-100    | ResNet-18+group   normalization |   FedAvg  | Pachinko Allocation | 100/500(ex/cli) |         500         |          10          |     20     |        SGD       |      0.1     |        -       |    1   |    >4000   |   44.7   |
|       NLP        |    Shakespeare   |       RNN (2 LSTM + 1 FC)       |   FedAvg  |  realistic patition |        　       |         715         |          10          |      4     |        SGD       |       1      |        -       |    1   |    >1200   |   56.9   |
|       NLP         |   StackOverflow  |       RNN (1 LSTM + 2 FC)       |   FedAvg  | Pachinko Allocation |        　       |        342477       |          50          |     16     |        SGD       | pow(10,-0.5) |        -       |    1   |    >1500   |   19.5   |

- Federated EMNIST-CNN-FedAvg (https://openreview.net/pdf?id=LkFG3lB13U5)
    - Patition Method: ‘Adaptive federated optimization’ (https://openreview.net/pdf?id=LkFG3lB13U5),  page 23, Appendix C.2
    - client_num_in_total: ‘Adaptive federated optimization’, page 23, Appendix C Dataset & Models, Table2
    - client_num_per_round: ‘Adaptive federated optimization’, page 6, Section 4, ‘Optimizer and hyperparameters’
    - batch_size: ‘Adaptive federated optimization’, page 27, Appendix D Experiment Hyperparameters, Table7
    - client_optimizer: ‘Adaptive federated optimization’, page 25, Appendix D.1, Paragraph 1
    - lr: ‘Adaptive federated optimization’, page 27, Appendix D.4, Table8
    - wd (learning rate decay):  ‘Adaptive federated optimization’, page34, Appendix E.6, Paragraph 2
    - epochs: ‘Adaptive federated optimization’, page34, Appendix E.6, Paragraph 1
    - comm_round:‘Adaptive federated optimization’, page28, Appendix E.1, figure 3
    - accuracy: ‘Adaptive federated optimization’, page 7, Section 5, Table1
- CIFAR-100 – ResNet18 -FedAvg
    - Patition Method: ‘Adaptive federated optimization’, page 23, Appendix C.1, Paragraph 3
    - Patition_alpha: ‘Adaptive federated optimization’, page 23, Appendix C.1, Paragraph 2
    - client_num_in_total: ‘Adaptive federated optimization’, page 23, Appendix C Dataset & Models, Table2
    - client_num_per_round: ‘Adaptive federated optimization’, page 6, Section 4, ‘Optimizer and hyperparameters’
    - batch_size: ‘Adaptive federated optimization’, page 27, Appendix D Experiment Hyperparameters, Table7
    - client_optimizer: ‘Adaptive federated optimization’, page 25, Appendix D.1, Paragraph 1
    - lr: ‘Adaptive federated optimization’, page 27, Appendix D.4, Table8
    - epochs: ‘Adaptive federated optimization’, page 6, Section 4, ‘Optimizer and hyperparameters’ 
    - comm_round: ‘Adaptive federated optimization’, page 7, Section 4, figure 1
    - accuracy: ‘Adaptive federated optimization’, page 7, Section 5, Table1
- Shakespeare – RNN – FedAvg
    - Patition Method: ‘Adaptive federated optimization’, page 23, Appendix C.3
    - client_num_in_total: ‘Adaptive federated optimization’, page 23, Appendix C Dataset & Models, Table2
    - client_num_per_round: ‘Adaptive federated optimization’, page 6, Section 4, ‘Optimizer and hyperparameters’
    - batch_size: ‘Adaptive federated optimization’, page 27, Appendix D Experiment Hyperparameters, Table7
    - client_optimizer: ‘Adaptive federated optimization’, page 25, Appendix D.1, Paragraph 1
    - lr: ‘Adaptive federated optimization’, page 27, Appendix D.4, Table8
    - epochs: ‘Adaptive federated optimization’, page 6, Section 4, ‘Optimizer and hyperparameters’ 
    - comm_round: ‘Adaptive federated optimization’, page 7, Section 4, figure 1
    - accuracy: ‘Adaptive federated optimization’, page 7, Section 5, Table1
- StackOverflow – RNN – FedAvg
    - Patition Method: ‘Adaptive federated optimization’, page 23, Appendix C.4, Paragraph 2
    - client_num_in_total: ‘Adaptive federated optimization’, page 25, Appendix C.4, Paragraph 1
    - client_num_per_round: ‘Adaptive federated optimization’, page 6, Section 4, ‘Optimizer and hyperparameters’
    - batch_size: ‘Adaptive federated optimization’, page 27, Appendix D Experiment Hyperparameters, Table7
    - client_optimizer: ‘Adaptive federated optimization’, page 25, Appendix D.1, Paragraph 1
    - lr: ‘Adaptive federated optimization’, page 27, Appendix D.4, Table8
    - epochs: ‘Adaptive federated optimization’, page 6, Section 4, ‘Optimizer and hyperparameters’ 
    - comm_round: ‘Adaptive federated optimization’, page 7, Section 4, figure 1
    - accuracy: ‘Adaptive federated optimization’, page 7, Section 5, Table1

### Benchmarking using modern DNNs
|     Data     |              Model              | Alg | # C | # C_p | bs | c_opt |      lr      |       wd       | e | round | IID acc | non-IID acc |
|:----------------:|:-------------------------------:|:---------------:|:-------------------:|:--------------------:|:----------:|:----------------:|:------------:|:--------------:|:------:|:----------:|:--------:|:--------:|
|     CIFAR10   |       ResNet-56       |   FedAvg  |        10         |          10          |      64     |        SGD       |       0.001      |        0.001      |    20   |    100   | 93.19  | 87.12 |
|     CIFAR100  |       ResNet-56       |   FedAvg  |        10       |          10          |     64     |        SGD       |        0.001      |        0.001      |    20   |    100   |   68.91 | 64.70 |
|     CINIC10  |        ResNet-56       |   FedAvg  |        10       |          10          |     64     |        SGD       |        0.001      |        0.001      |    20   |    100   |  82.57  | 73.49 | 
|     CIFAR10   |       MobileNet       |   FedAvg  |        10         |          10          |      64     |        SGD       |          0.001      |        0.001      |    20   |    100   | 91.12 | 86.32 |
|     CIFAR100  |       MobileNet       |   FedAvg  |        10       |          10          |     64     |        SGD       |        0.001      |        0.001      |    20   |    100   |  55.12 | 53.54 | 
|     CINIC10  |        MobileNet       |   FedAvg  |        10       |          10          |     64     |        SGD       |        0.001      |        0.001      |    20   |    100   |   79.95 | 71.23|

Note: Non-IID distribution is set using LDA ( LDA = Latent Dirichlet Allocation) with alpha = 0.5; #C stands for client_num_in_total; #C_p stands for client_num_per_round; bs = batch size; c_opt = client optimizer.
