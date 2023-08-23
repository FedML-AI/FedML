# base
FROM fedml/fedml:latest-torch1.13.1-cuda11.6-cudnn8-devel

# set the github runner version
ARG RUNNER_VERSION="2.304.0"

# update the base packages and add a non-sudo user
#RUN apt-get update -y && apt-get upgrade -y && useradd -m docker

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git

# cd into the user directory, download and unzip the github actions runner
WORKDIR /home/fedml

RUN  mkdir actions-runner && cd actions-runner \
    &&  curl -O -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
    && tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

# install some additional dependencies
#RUN chown -R docker ~docker && /home/docker/actions-runner/bin/installdependencies.sh

# copy over the start.sh script
COPY start.sh start.sh

# make the script executable
RUN chmod +x start.sh

RUN cp -f /usr/bin/python /usr/bin/python-backup && ln -s /usr/bin/python3 python

RUN pip install scikit-learn

RUN pip install tensorflow && pip install tensorflow_datasets && pip install jax[cpu] && pip install dm-haiku && pip install optax && pip install jaxlib

# since the config and run script for actions are not allowed to be run by root,
# set the user to "docker" so all subsequent commands are run as the docker user
#USER docker

ENV REPO=FedML-AI/FedML ACCESS_TOKEN=1

# set the entrypoint to the start.sh script
CMD ./start.sh ${REPO} ${ACCESS_TOKEN}