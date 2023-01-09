ARG VERSION=dev
ARG IS_BUILDING_GPU_IMAGE=0
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:22.01-py3
FROM ${BASE_IMAGE}

ADD ./devops/scripts/runner.sh ./fedml/runner.sh

ADD ./devops/scripts/requirements.txt ./fedml/requirements.txt

RUN chmod a+x ./fedml/runner.sh
RUN echo "Updating..."

RUN pip3 install -r ./fedml/requirements.txt

COPY ./python ./fedml/fedml-pip
WORKDIR ./fedml/fedml-pip
RUN pip3 install -e ./

WORKDIR /fedml

ENV ACCOUNT_ID=0 FEDML_VERSION=${VERSION} CLIENT_DEVICE_ID=0 CLIENT_OS_NAME=linux
