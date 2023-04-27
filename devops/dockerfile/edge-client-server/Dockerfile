ARG VERSION=dev
ARG IS_BUILDING_GPU_IMAGE=0
ARG BASE_IMAGE=fedml/fedml:latest-torch1.13.1-cuda11.6-cudnn8-devel
FROM ${BASE_IMAGE}

ADD ./devops/scripts/edge-client-server.sh ./fedml/edge-client-server.sh

RUN chmod a+x ./fedml/edge-client-server.sh
RUN echo "Updating..."

WORKDIR /fedml

ENV ROLE=client ACCOUNT_ID=0

CMD ./edge-client-server.sh ${ROLE} ${ACCOUNT_ID}