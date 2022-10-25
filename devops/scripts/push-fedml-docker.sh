#!/bin/bash

export FEDML_VERSION=`cat python/setup.py |grep version= |awk -F'=' '{print $2}' |awk -F',' '{print $1}'|awk -F'"' '{print $2}'`
docker push fedml/fedml:latest-torch1.12.1-cuda11.3-cudnn8-devel