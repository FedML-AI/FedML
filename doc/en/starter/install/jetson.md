# FedML Installation on NVIDIA Jetson Devices

## Run FedML with Docker (Recommended)
- Pull FedML RPI docker image
```
docker pull fedml/fedml:nvidia-jetson-l4t-ml-r32.6.1-py3
```

- Run Docker with "fedml login"
```
docker run -t -i --runtime nvidia fedml/fedml:nvidia-jetson-l4t-ml-r32.6.1-py3 /bin/bash

root@8bc0de2ce0e0:/usr/src/app# fedml login 299

```

## Install with pip
This method is only recommended to those who don't want to use docker. 
fedml needs to be installed without dependencies because Pytorch is not available in pip on Jetson.
1. install Pytorch using [python wheels](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-11-now-available/72048)
2. install rest of the [dependencies](https://github.com/FedML-AI/FedML/blob/d9bc5fdfe5b4b6d9b59139d3f017702d644ce040/python/setup.py#L20) with pip3 except torch and torchvision:
3. install fedml without dependencies since they are installed manually.
```
pip3 install fedml --no-dependencies
```
### Issues and solutions:
1. pip3 install h5py related error:
```
sudo apt-get install subversion
ln -s /usr/include/locale.h /usr/include/xlocale.h
sudo apt-get install libhdf5-serial-dev
```

2. pip3 install sklearn related error:
```
sudo apt-get install build-essential libatlas-base-dev gfortran
```
