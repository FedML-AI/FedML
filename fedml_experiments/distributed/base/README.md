## 1. FedML: A Flexible Distributed Machine Learning Library for Novel Learning Algorithms and Models
http://fedml.ai

Python package homepage: http://pypi.org/project/fedml-core

## 2. Environmental Setups

### 2.1 Hardware Requirements
![multi-gpu-server](./../docs/images/multi-gpu-topo1.png)

The computing architecture is comprised of \
N compute nodes, each compute node is a multi-GPU server node (e.g., 8 x NVIDIA V100). \
A head node (login, testing) \
A centralized fault-tolerant file-server (NFS). In machine learning setting, this is used to share large-scale dataset among compute nodes. \

If you need FedML to support a physical architecture that is different from the above topology, please contact http://chaoyanghe.com.

### 2.2 Core library Introduction

Code implementation is based on PyTorch 1.4.0, MPI4Py 3.0.3 (https://pypi.org/project/mpi4py), and Python 3.7.4.

The experiment tracking platform is supported by Weights and Bias: https://www.wandb.com/

### **- FedML**
FedML: A Flexible Distributed Machine Learning Library for Novel Learning Algorithms and Models.
> pip install fedml-core

### 2.3 Software Configuration
Here is a step-by-step configuration to help you quickly set up a multi-GPU computing environment.
### **- Conda**

https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html

### **- PyTorch**

> conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

### **- MPI4py**
> conda install -c anaconda mpi4py

### **- Weights and Bias**
> pip install --upgrade wandb

### **- NFS (Network File System) Configuration**
Please google related installment instructions according to the OS version of your server.

### **- change SSH configuration for your cluster**

- On your local computer (MAC/Windows), generate the public key:
> mkdir ~/.ssh
> ls ~/.ssh
> ssh-keygen -t rsa
> vim ~/.ssh/id_rsa.pub

- Login to the server-side:
> ssh chaoyang@gpumaster-scip.usc.edu

- modify the "authorized_keys"
> vim ~/.ssh/authorized_keys


Paste the string in "id_rsa.pub" file on your local computer to the server side "authorized_keys" file, and save the authorized_keys
> chmod 700 ~/.ssh/
> chmod 600 ~/.ssh/authorized_keys


- login out and login again, you will find you don't need to input the passwords anymore.

For other nodes on your server, use a similar method to configure the SSH.

### **- config MPI host file**
Modify the hostname list in "mpi_host_file" to correspond to your actual physical network topology.
An example: Let us assume a network has a management node and four compute nodes (hostname: node1, node2, node3, node4).
If you want use node1 and node2 to run our program, the "mpi_host_file" should be:
> node1 \
> node2 \
> node3


## 3. download dataset



## 4. Running Experiments 

#### CIFAR10
train on IID dataset 
```
sh run_fedavg_distributed_pytorch.sh
```