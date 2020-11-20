# Benchmarking Experimental Results
Please visit the following link to check the latest benchmark experimental results.

https://app.wandb.ai/automl/fedml/reports/FedML-Benchmark-Experimental-Results--VmlldzoxODE2NTU

### Hyperparameters

|     Data Set     |              Model              | Alogrithm |   Partition Method  | Partition Alpha | client_num_in_total | client_num_per_round | batch_size | client_optimizer |      lr      |       wd       | epochs | comm_round | accuracy |
|:----------------:|:-------------------------------:|:---------:|:-------------------:|:---------------:|:-------------------:|:--------------------:|:----------:|:----------------:|:------------:|:--------------:|:------:|:----------:|:--------:|
|       MNIST      |       Logistic Regression       |   FedAvg  |      Power Law      |        　       |         1000        |          10          |     10     |        SGD       |     0.03     |        -       |    1   |    >100    |    >75   |
| Federated EMNIST |       Logistic Regression       |   FedAvg  |  realistic patition |        　       |         200         |          10          |     10     |        SGD       |     0.003    |        -       |    1   |    >200    |   10~40  |
|  Synthetic(α,β） |       Logistic Regression       |   FedAvg  |      Power Law      |        　       |          30         |          10          |     10     |        SGD       |     0.01     |        -       |    1   |    >200    |    >60   |
| Federated EMNIST |       CNN (2 Conv + 2 FC)       |   FedAvg  |  realistic patition |        　       |         3400        |          10          |     20     |        SGD       |      0.1     | 0.1/500 rounds |   100  |    >1500   |   84.9   |
|     CIFAR-100    | ResNet-18+group   normalization |   FedAvg  | Pachinko Allocation | 100/500(ex/cli) |         500         |          10          |     20     |        SGD       |      0.1     |        -       |    1   |    >4000   |   44.7   |
|    Shakespeare   |       RNN (2 LSTM + 1 FC)       |   FedAvg  |  realistic patition |        　       |         715         |          10          |      4     |        SGD       |       1      |        -       |    1   |    >1200   |   56.9   |
|   StackOverflow  |       RNN (1 LSTM + 2 FC)       |   FedAvg  |  realistic patition |        　       |        342477       |          50          |     16     |        SGD       | pow(10,-0.5) |        -       |    1   |    >1500   |   19.5   |
​
> for Synthetic(α,β), (α,β) is chosen from (0,0), (0.5,0.5), (1,1)
### Reference Lists
We refer the hyper-parameters from many top-tier ML conferences. Please check details of our reference hyperparameters as follows.

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
- Federated EMNIST-CNN-FedAvg
    - Patition Method: ‘Adaptive federated optimization’,  page 23, Appendix C.2
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
