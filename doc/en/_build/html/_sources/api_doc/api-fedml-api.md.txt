# FedML APIs (high-level) 

FedML-API is built based on [FedML-Core](api-core.md).
With the help of FedML-core, new algorithms in distributed version can be easily implemented by adopting the worker-oriented programming interface, 
which is a novel design pattern for flexible distributed computing.
Such a distributed computing paradigm is essential for scenarios in which large DNN training cannot be handled by standalone simulation due to GPU memory and training time constraints.
We specifically point out that this distributed computing design is not only used for FL, but it can also be used for conventional in-cluster large-scale distributed training (e.g., training modern neural architectures like CNNs or transformers).
FedML-API also suggests a machine learning system practice that separates the implementations of models, datasets, and algorithms.
This practice can enable code reuse and also fair comparison, avoiding statistical or system-level gaps among algorithms led by non-trivial implementation differences.
Another benefit is that FL applications can develop more models and submit more realistic datasets without the need to understand the details of different distributed optimization algorithms.
We hope that researchers in diverse FL applications can contribute more valuable models and realistic datasets to our community.