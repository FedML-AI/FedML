ARG VERSION=dev
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

ENV FEDML_REDIS_ADDR="local" FEDML_REDIS_PORT=6379 FEDML_REDIS_PASSWORD="fedml_default" \
    FEDML_END_POINT_ID=0 FEDML_MODEL_ID=0 \
    FEDML_MODEL_NAME="model" FEDML_MODEL_VERSION="v1" \
    FEDML_INFER_URL="infer" FEDML_CONFIG_VERSION="release" \
    FEDML_INFER_PORT=5001

CMD python3 ${FEDML_PIP_HOME}/cli/model_deployment/device_model_inference_entry.py \
    -rp ${FEDML_REDIS_ADDR} -ra ${FEDML_REDIS_PORT} -rpw ${FEDML_REDIS_PASSWORD} \
    -ep ${FEDML_END_POINT_ID} -mi ${FEDML_MODEL_ID} \
    -mn ${FEDML_MODEL_NAME} -mv ${FEDML_MODEL_VERSION} \
    -iu ${FEDML_INFER_URL} -cv ${FEDML_CONFIG_VERSION} \
    -ip ${FEDML_INFER_PORT};./runner.sh