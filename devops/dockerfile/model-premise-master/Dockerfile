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

RUN rm -f /usr/share/keyrings/redis-archive-keyring.gpg;curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list;  \
    apt-get update; service redis-server stop;apt-get install -y redis

WORKDIR /fedml

ENV ACCOUNT_ID=0 FEDML_VERSION=${VERSION} SERVER_DEVICE_ID=0 SERVER_OS_NAME=linux INFER_HOST="127.0.0.1" \
    FEDML_REDIS_ADDR="127.0.0.1" FEDML_REDIS_PORT=6379 FEDML_REDIS_PASSWORD="fedml_default"

CMD fedml model device login ${ACCOUNT_ID} -v ${FEDML_VERSION} -p -m \
    -ih ${INFER_HOST} -id ${SERVER_DEVICE_ID} -os ${SERVER_OS_NAME} \
    -ra ${FEDML_REDIS_ADDR} -rp ${FEDML_REDIS_PORT} -rpw ${FEDML_REDIS_PASSWORD};./runner.sh