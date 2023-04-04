#!/bin/bash

version=$1
pwd=`pwd`

docker login

docker tag public.ecr.aws/x6k8q1x9/fedml-inference-backend:latest fedml/fedml-inference-backend:latest
docker push fedml/fedml-inference-backend:latest

docker tag public.ecr.aws/x6k8q1x9/fedml-inference-ingress:${version} fedml/fedml-inference-ingress:${version}
docker push fedml/fedml-inference-ingress:${version}

docker tag public.ecr.aws/x6k8q1x9/fedml-model-premise-master:${version} fedml/fedml-model-premise-master:${version}
docker push fedml/fedml-model-premise-master:${version}

docker tag public.ecr.aws/x6k8q1x9/fedml-model-premise-slave:${version} fedml/fedml-model-premise-slave:${version}
docker push fedml/fedml-model-premise-slave:${version}
