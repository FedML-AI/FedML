#!/bin/bash

export FEDML_VERSION=`cat python/setup.py |grep version= |awk -F'=' '{print $2}' |awk -F',' '{print $1}'|awk -F'"' '{print $2}'`
docker push fedml/fedml:latest-torch1.12.1-cuda11.3-cudnn8-devel
docker push fedml/fedml:${FEDML_VERSION}-torch1.12.1-cuda11.3-cudnn8-devel

docker push fedml/fedml:latest-torch1.12.1-cuda11.3-cudnn8-devel-arm64
docker push fedml/fedml:${FEDML_VERSION}-torch1.12.1-cuda11.3-cudnn8-devel-arm64

docker push fedml/fedml:latest-nvidia-jetson-l4t-ml-r32.6.1-py3
docker push fedml/fedml:${FEDML_VERSION}-nvidia-jetson-l4t-ml-r32.6.1-py3

docker push fedml/fedml:latest-raspberrypi4-32-py37
docker push fedml/fedml:${FEDML_VERSION}-raspberrypi4-32-py37

docker push fedml/fedml:latest-raspberrypi4-64-py37
docker push fedml/fedml:${FEDML_VERSION}-raspberrypi4-64-py37