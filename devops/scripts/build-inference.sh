#!/bin/bash

version=$1
pwd=`pwd`

docker login --username AWS --password $(aws ecr-public get-login-password --region us-east-1) public.ecr.aws

docker build -f ./devops/dockerfile/model-inference-base/Dockerfile -t public.ecr.aws/x6k8q1x9/fedml-inference-base-image:latest .
docker push public.ecr.aws/x6k8q1x9/fedml-inference-base-image:latest

docker build -f ./devops/dockerfile/model-inference-ingress/Dockerfile -t public.ecr.aws/x6k8q1x9/fedml-model-inference-ingress:${version} .
docker push public.ecr.aws/x6k8q1x9/fedml-model-inference-ingress:${version}

docker build -f ./devops/dockerfile/model-premise-master/Dockerfile -t public.ecr.aws/x6k8q1x9/fedml-model-premise-master:${version} .
docker push public.ecr.aws/x6k8q1x9/fedml-model-premise-master:${version}

docker build -f ./devops/dockerfile/model-premise-slave/Dockerfile -t public.ecr.aws/x6k8q1x9/fedml-model-premise-slave:${version} .
docker push public.ecr.aws/x6k8q1x9/fedml-model-premise-slave:${version}
