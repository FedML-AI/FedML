ARG VERSION=test
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

WORKDIR ./fedml

ENV MODE=normal FEDML_VERSION=${VERSION} ACCOUNT_ID=0 SERVER_DEVICE_ID=0 \
    FEDML_PACKAGE_NAME=package FEDML_PACKAGE_URL=s3_url \
    FEDML_RUNNER_CMD=3dsad

CMD fedml login ${ACCOUNT_ID} -v ${FEDML_VERSION} -s -r cloud_server -rc ${FEDML_RUNNER_CMD} -id ${SERVER_DEVICE_ID}; ./runner.sh