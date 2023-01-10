ARG VERSION=dev
ARG IS_BUILDING_GPU_IMAGE=0
ARG BASE_IMAGE=public.ecr.aws/x6k8q1x9/fedml-device-image:base
ARG FEDML_PIP_HOME=/usr/local/lib/python3.7/dist-packages/fedml
FROM ${BASE_IMAGE}

ADD ./devops/scripts/runner.sh ./fedml/runner.sh

ADD ./devops/scripts/requirements.txt ./fedml/requirements.txt

RUN chmod a+x ./fedml/runner.sh
RUN echo "Updating..."

RUN pip3 install -r ./fedml/requirements.txt

COPY ./python ./fedml/fedml-pip
COPY ./python/fedml ${FEDML_PIP_HOME}
WORKDIR ./fedml/fedml-pip
RUN pip3 install -e ./

WORKDIR /fedml

ENV ACCOUNT_ID=0 FEDML_VERSION=${VERSION} CLIENT_DEVICE_ID=0 CLIENT_OS_NAME=linux INFER_HOST="127.0.0.1"

CMD fedml model device login ${ACCOUNT_ID} -v ${FEDML_VERSION} -p \
    -id ${CLIENT_DEVICE_ID} -os ${CLIENT_OS_NAME} -ih ${INFER_HOST}; ./runner.sh