ARG VERSION=dev
ARG IS_BUILDING_GPU_IMAGE=0
ARG BASE_IMAGE=public.ecr.aws/x6k8q1x9/fedml-device-image:base
FROM ${BASE_IMAGE}

ADD ./devops/scripts/runner.sh ./fedml/runner.sh

ADD ./devops/scripts/requirements.txt ./fedml/requirements.txt

RUN chmod a+x ./fedml/runner.sh
RUN echo "Updating..."

RUN pip3 install -r ./fedml/requirements.txt

COPY ./python ./fedml/fedml-pip
WORKDIR ./fedml/fedml-pip
RUN pip3 install -e ./
#RUN pip3 install -e '.[tensorflow]'
#RUN pip3 install -e '.[jax]'
#RUN pip3 install -e '.[mxnet]'

WORKDIR /fedml

ENV ACCOUNT_ID=0 FEDML_VERSION=${VERSION} CLIENT_DEVICE_ID=0 CLIENT_OS_NAME=linux

CMD fedml login ${ACCOUNT_ID} -v ${FEDML_VERSION} -c -id ${CLIENT_DEVICE_ID} -os ${CLIENT_OS_NAME}; ./runner.sh