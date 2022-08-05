ARG VERSION=dev
ARG BASE_IMAGE=public.ecr.aws/x6k8q1x9/fedml-device-image:base
FROM ${BASE_IMAGE}

ADD ./devops/scripts/copy-fedml-package.sh ./fedml/copy-fedml-package.sh
ADD ./devops/scripts/runner.sh ./fedml/runner.sh

COPY ./python/fedml ./fedml/fedml-pip
RUN chmod a+x ./fedml/copy-fedml-package.sh
RUN bash ./fedml/copy-fedml-package.sh

RUN chmod a+x ./fedml/runner.sh

WORKDIR ./fedml

ENV ACCOUNT_ID=0 FEDML_VERSION=${VERSION} SERVER_DEVICE_ID=0 SERVER_OS_NAME=linux

CMD fedml login ${ACCOUNT_ID} -v ${FEDML_VERSION} -s -id ${SERVER_DEVICE_ID} -os ${SERVER_OS_NAME};./runner.sh