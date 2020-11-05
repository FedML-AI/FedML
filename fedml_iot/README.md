# FedML-IoT: Federated Learning on IoT Devices (Raspberry Pi 4)

Our FedML architecture design can smoothly transplant the distributed computing code to the IoT platform. 

Currently, we support two IoT devices: 

###1. Raspberry PI 4 (Edge CPU Computing - ARMv7l)
<img src=https://github.com/FedML-AI/FedML/blob/master/docs/image/raspberry_pi.png width="35%">

You can buy RPI 4 devices here: https://www.raspberrypi.org/products/raspberry-pi-4-desktop-kit/?resellerType=home

###1. NVIDIA Jetson Nano (Edge GPU Computing)
<img src=https://github.com/FedML-AI/FedML/blob/master/docs/image/nvidia-jetson-nano.png width="35%">

About NVIDIA Jetson Nano:
https://developer.nvidia.com/embedded/jetson-nano-developer-kit

NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

You can buy Jetson Nano here:
https://developer.nvidia.com/buy-jetson?product=jetson_tx2&location=US


# Installation 
### Install FedML + PyTorch 1.4 on Raspberry Pi 4
The following commands are running on Raspberry Pi 4
```
cd /home/pi
mkdir sourcecode
git clone https://github.com/FedML-AI/FedML.git
cd FedML/fedml_iot
```
After the above commands, please follow the script `install-arm.sh`.

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

### Install FedML + PyTorch 1.4 on Raspberry Pi 4
We are still testing the script on NVIDIA Jetson Nano, please stay turned.

# Log Tracking
Our log tracking platform is wandb.com. Please register your own ID and login as follows.
```
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
```

# Launch FedML-Mobile Server
FedML-IoT reuses the FedML-Mobile server code. So you can launch the server according to the guidance at `fedml_mobile/server/executor`.
Please change the IP address of the server. Here, we assume the server IP is `127.0.0.1`.
```
python app.py
```
The default training is MNIST (dataset) + FedAvg (optimizer) + LR (model).
You can customize `app.py` as your need.



# Launch FedML-IoT Client
Here we assume you have FOUR IoT devices. Then you can run script in each one as follows.
If your RPI device has more memory, you can run multiple `fedavg_rpi_client.py` processes in a single device.
```
cd ./raspberry_pi/fedavg
python fedavg_rpi_client.py --server_ip http://127.0.0.1:5000 --client_uuid '0'
python fedavg_rpi_client.py --server_ip http://127.0.0.1:5000 --client_uuid '1'
python fedavg_rpi_client.py --server_ip http://127.0.0.1:5000 --client_uuid '2'
python fedavg_rpi_client.py --server_ip http://127.0.0.1:5000 --client_uuid '3'
```
Note please change IP and other configuration according to your local environment.
