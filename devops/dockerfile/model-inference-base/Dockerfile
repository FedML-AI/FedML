ARG VERSION=dev
ARG IS_BUILDING_GPU_IMAGE=0
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:22.01-py3
FROM ${BASE_IMAGE}

ADD ./devops/scripts/runner.sh ./fedml/runner.sh

ADD ./devops/scripts/requirements.txt ./fedml/requirements.txt

RUN chmod a+x ./fedml/runner.sh
RUN echo "Updating..."
