ARG VERSION=dev
ARG IS_BUILDING_GPU_IMAGE=0
ARG BASE_IMAGE=public.ecr.aws/x6k8q1x9/fedml-inference-base-image:latest
ARG FEDML_PIP_HOME=/usr/local/lib/python3.8/dist-packages/fedml
FROM ${BASE_IMAGE}

ADD ./devops/scripts/runner.sh ./fedml/runner.sh

ADD ./devops/scripts/requirements.txt ./fedml/requirements.txt

RUN chmod a+x ./fedml/runner.sh
RUN echo "Updating..."

RUN pip3 install -r ./fedml/requirements.txt

RUN python3 -m pip install --upgrade pip
RUN pip3 install fedml

COPY ./python ./fedml/fedml-pip
COPY ./python/fedml ${FEDML_PIP_HOME}
WORKDIR ./fedml/fedml-pip
RUN pip3 install -e ./

WORKDIR /fedml

ENV ACCOUNT_ID=0 FEDML_VERSION=${VERSION} CLIENT_DEVICE_ID=0 CLIENT_OS_NAME=linux

CMD fedml model device login ${ACCOUNT_ID} -v ${FEDML_VERSION} -p -id ${CLIENT_DEVICE_ID} -os ${CLIENT_OS_NAME}; ./runner.sh