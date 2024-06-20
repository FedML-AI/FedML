# base
ARG BASE_IMAGE=python:3.11

FROM ${BASE_IMAGE}

# set the github runner version
ARG RUNNER_VERSION="2.317.0"
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
# since the config and run script for actions are not allowed to be run by root,
# set the user to "docker" so all subsequent commands are run as the docker user
#USER docker

RUN git clone https://github.com/Qigemingziba/FedML.git
RUN cd FedML && git pull && git checkout dev/v0.7.0 && cd python && pip3 install -e ./ 
ENV REPO=Qigemingziba/FedML ACCESS_TOKEN=AGMK3P4W5EM5PXNYTZXXIMTGNF4MW

# set the entrypoint to the start.sh script
CMD ./start.sh ${REPO} ${ACCESS_TOKEN} 


