ARG VERSION=dev
ARG IS_BUILDING_GPU_IMAGE=0
ARG BASE_IMAGE=public.ecr.aws/x6k8q1x9/fedml-device-image:base
FROM ${BASE_IMAGE}

ADD ./devops/scripts/copy-fedml-package.sh ./fedml/copy-fedml-package.sh
ADD ./devops/scripts/runner.sh ./fedml/runner.sh

COPY ./python/fedml ./fedml/fedml-pip
RUN chmod a+x ./fedml/copy-fedml-package.sh
RUN bash ./fedml/copy-fedml-package.sh

RUN chmod a+x ./fedml/runner.sh

WORKDIR ./fedml

ENV MODE=normal FEDML_VERSION=${VERSION} ACCOUNT_ID=0 SERVER_AGENT_ID=0 \
    AWS_IAM_ACCESS_ID=0 \
    AWS_IAM_ACCESS_KEY=0 \
    AWS_REGION=0

CMD ./set-aws-credentials.sh ${AWS_IAM_ACCESS_ID} ${AWS_IAM_ACCESS_KEY} ${AWS_REGION};fedml login ${ACCOUNT_ID} -v ${FEDML_VERSION} -s -r cloud_agent -id ${SERVER_AGENT_ID};./runner.sh