ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:22.01-py3
FROM ${BASE_IMAGE}

ENV DATA_CACHE_FOLDER=""
VOLUME [ DATA_CACHE_FOLDER ]

CMD /bin/bash ${BOOTSTRAP_DIR};\
tritonserver --model-repository=${DATA_CACHE_FOLDER};