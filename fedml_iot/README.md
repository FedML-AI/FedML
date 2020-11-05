# FedML on IoT Devices (Raspberry Pi 4)
<img src=https://github.com/FedML-AI/FedML/blob/master/docs/image/raspberry_pi.png width="35%">

Our FedML architecture design can smoothly transplant the distributed computing code to the IoT platform. Here we use Raspberry PI to demonstrate our idea. 

You can buy RPI 4 devices here: https://www.raspberrypi.org/products/raspberry-pi-4-desktop-kit/?resellerType=home


# Installation 
### Install FedML + PyTorch 1.4 in Raspberry Pi 4
Please follow the script `install-arm.sh`.

Note: This script has been tested on Raspberry Pi 4 (RPI 4). We welcome users to contribute more scripts for new IoT platforms. 

We currently support PyTorch 1.4 for Raspberry Pi 4.
If you need newer verion of PyTorch, please compile the wheel file using method here: https://nmilosev.svbtle.com/compling-arm-stuff-without-an-arm-board-build-pytorch-for-the-raspberry-pi

To make sure your PyTorch + RPI environment is ready, please have a check as follows.
```
$ python

>>> import torch
>>> import torchvision
>>> import cv2
```

# Log Tracking
Our log tracking platform is wandb.com. Please register your own ID and login as follows.
```
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
```

# Launch FedML-Mobile Server
FedML-IoT reuses the FedML-Mobile server code. So you can launch the server according to the guidance at `fedml_mobile/server/executor`.
```
python app.py
```
The default training is MNIST (dataset) + FedAvg (optimizer) + LR (model).
You can customize `app.py` as your need.

Assume the server IP is `127.0.0.1`.

# Launch FedML-IoT Client
Here we assume you have FOUR IoT devices. Then you can run script in each one as follows.
```
python fedml_iot_client --client_uuid '0'
python fedml_iot_client --client_uuid '1'
python fedml_iot_client --client_uuid '2'
python fedml_iot_client --client_uuid '3'
```

Note please change IP and other configuration according to your local environment.

# The training 
