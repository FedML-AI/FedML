#!/bin/bash

version=local
pwd=`pwd`

docker login --username AWS --password $(aws ecr-public get-login-password --region us-east-1) public.ecr.aws
docker build --network=host -f ./devops/dockerfile/device-image/Dockerfile-Dev -t public.ecr.aws/x6k8q1x9/fedml-device-image:${version} .
#docker push public.ecr.aws/x6k8q1x9/fedml-device-image:${version}

#docker build -f ./devops/dockerfile/server-agent/Dockerfile-Dev -t public.ecr.aws/x6k8q1x9/fedml-server-agent:${version} .
#docker push public.ecr.aws/x6k8q1x9/fedml-server-agent:${version}