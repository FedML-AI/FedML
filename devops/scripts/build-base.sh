#!/bin/bash

version=base
pwd=`pwd`

docker login --username AWS --password $(aws ecr-public get-login-password --region us-east-1) public.ecr.aws
docker build -f ./devops/dockerfile/device-image/Dockerfile-Base -t public.ecr.aws/x6k8q1x9/fedml-device-image:${version} .
docker push public.ecr.aws/x6k8q1x9/fedml-device-image:${version}
