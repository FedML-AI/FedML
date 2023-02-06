ARG VERSION=dev
ARG IS_BUILDING_GPU_IMAGE=0
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:22.01-py3
FROM ${BASE_IMAGE}

ADD ./devops/scripts/runner.sh ./fedml/runner.sh

ADD ./devops/scripts/requirements.txt ./fedml/requirements.txt

RUN chmod a+x ./fedml/runner.sh
RUN pip install transformers

ENV FEDML_MODEL_SERVING_REPO_SCAN_INTERVAL=3 \
    FEDML_MODEL_SERVING_REPO_PATH=/home/fedml/fedml-client/fedml/models_serving

CMD tritonserver --model-control-mode=poll \
    --strict-model-config=false \
    --backend-config=onnx,default-max-batch-size=4 \
    --repository-poll-secs=${FEDML_MODEL_SERVING_REPO_SCAN_INTERVAL} \
    --model-repository=${FEDML_MODEL_SERVING_REPO_PATH}