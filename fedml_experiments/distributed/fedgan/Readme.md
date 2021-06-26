## Abstract
This is the toy example for the federated learning based gan. 


FedGAN Api is created by Lei Gao, Tuo Zhang, Qi Chang ang Zhennan Yan. For any questions during the using, please contact us via the FedML Slack.

## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Run Experiments
The toy examlpes is training a GAN with the MNIST dataset.
Note please tune hyper-parameters accordingly. 
You can refer the hyper-parameter suggestions at `FedML/benchmark/README.md`), but this may not be the optimal.

## Usage
Open fedgan under the fedml_experiments/distributed. Run the following command in terminal.
``
sh run_FedGAN.sh 3 3 gan 50 1 64 0.001 
``
The created image will be stored in the samples folder.
