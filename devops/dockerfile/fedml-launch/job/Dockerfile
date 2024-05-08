# Image Name: fedml/fedml-launch-job:cu11.6
ARG BASE_IMAGE=fedml/fedml:latest-torch1.13.1-cuda11.6-cudnn8-devel
FROM ${BASE_IMAGE}

ARG HOME_DIR=/home/fedml
## Only Modify if you want to use a different version of FedML
RUN mkdir -p ${HOME_DIR}/fedml-pip
COPY ./python ${HOME_DIR}/fedml-pip
WORKDIR ${HOME_DIR}/fedml-pip
RUN pip3 install ./

# 1. Specify Bootrap path (If any)
ENV BOOTSTRAP_DIR=""
    
# 2. MOUNT User's Local Folder (If any)
ENV DATA_FOLDER=""
VOLUME [ DATA_FOLDER ]

# 3. MOUNT Fedml Home Folder
VOLUME ["/home/fedml/launch"]

# 4. Enter the entrypoint
WORKDIR ${HOME_DIR}/launch
ENV MAIN_ENTRY=""

# if bootstrap dir is not empty, then run the bootstrap script
CMD /bin/bash ${BOOTSTRAP_DIR}; python3 ${MAIN_ENTRY}
