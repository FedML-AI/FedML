ARG VERSION=dev
ARG IS_BUILDING_GPU_IMAGE=0
ARG BASE_IMAGE=public.ecr.aws/x6k8q1x9/fedml-device-image:base
FROM ${BASE_IMAGE}

ADD ./devops/scripts/copy-fedml-package.sh ./fedml/copy-fedml-package.sh
ADD ./devops/scripts/runner.sh ./fedml/runner.sh
ADD ./devops/scripts/entry-point.sh ./fedml/entry-point.sh

COPY ./python/fedml ./fedml/fedml-pip
RUN chmod a+x ./fedml/copy-fedml-package.sh
RUN bash ./fedml/copy-fedml-package.sh

RUN chmod a+x ./fedml/runner.sh
RUN chmod a+x ./fedml/entry-point.sh
RUN echo "Updating..."

WORKDIR ./fedml

ENV ACCOUNT_ID=0 FEDML_VERSION=${VERSION} CLIENT_DEVICE_ID=0 CLIENT_OS_NAME=linux

CMD fedml login ${ACCOUNT_ID} -v ${FEDML_VERSION} -c -id ${CLIENT_DEVICE_ID} -os ${CLIENT_OS_NAME}; ./runner.sh