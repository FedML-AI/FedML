# nerdctl build -t fedml/fedml-k8s-agent:202412271540 -f /root/fedml-k8s-agent/Dockerfile .
# Use ubuntu 22.04 as base image
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/root/miniconda3/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    iproute2 \
    ca-certificates \
    gnupg \
    sudo \
    lsb-release \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Redis
RUN curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" \
    | tee /etc/apt/sources.list.d/redis.list \
    && apt-get update \
    && apt-get install -y redis \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /root/miniconda3 \
    && rm /tmp/miniconda.sh

# Initialize conda and install FedML
RUN conda init bash \
    && . /root/.bashrc \
    && conda create -y -n fedml python=3.10 \
    && conda activate fedml \
    && pip install --upgrade gevent multiprocess \
    && pip install "git+https://github.com/FedML-AI/FedML.git@k8s/dev/v0.7.0#egg=fedml&subdirectory=python"

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set default environment variables (these can be overridden in K8s deployment)
ENV FEDML_API_KEY=""
ENV FEDML_ENV="production"
ENV FEDML_PROVIDER="false"
ENV FEDML_DEVICE_ID=""
ENV INFERENCE_GATEWAY_PORT=""
ENV INFERENCE_PROXY_PORT=""
ENV FEDML_CONNECTION_TYPE=""

ENTRYPOINT ["/entrypoint.sh"]