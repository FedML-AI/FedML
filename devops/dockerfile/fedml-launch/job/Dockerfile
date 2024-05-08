# Image Name: fedml/fedml-launch-job:cu11.6
ARG BASE_IMAGE=fedml/fedml:latest-torch1.13.1-cuda11.6-cudnn8-devel
FROM ${BASE_IMAGE}

ARG HOME_DIR=/home/fedml
## Only Modify if you want to use a different version of FedML
RUN mkdir -p ${HOME_DIR}/fedml-pip
COPY ./python ${HOME_DIR}/fedml-pip
WORKDIR ${HOME_DIR}/fedml-pip
RUN pip3 install ./
    
# 2. MOUNT User's Local Folder (If any)
ENV DATA_FOLDER=""
VOLUME [ DATA_FOLDER ]

# 3. MOUNT Fedml Home Folder
VOLUME ["/home/fedml/launch"]

# 4. Enter the bootstrap and job entrypoints
WORKDIR ${HOME_DIR}/launch
ENV BOOTSTRAP_SCRIPT="fedml_bootstrap_generated.sh"
ENV MAIN_ENTRY="fedml_job_entry_pack.sh'"

# Run Bootstrap Script and Main Entry File
CMD /bin/bash ${BOOTSTRAP_SCRIPT}; /bin/bash ${MAIN_ENTRY}