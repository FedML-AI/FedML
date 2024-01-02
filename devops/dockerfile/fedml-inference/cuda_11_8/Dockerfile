ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:22.12-py3
FROM ${BASE_IMAGE}

## Only Modify if you want to use a different version of FedML
RUN mkdir -p /fedml-pip
ADD ./python ./fedml-pip
WORKDIR ./fedml-pip
RUN pip3 install -e ./

# 1. Specify Bootrap path (If any)
ENV BOOTSTRAP_DIR=""
    
# 2. MOUNT User's Local Folder (If any)
ENV DATA_CACHE_FOLDER=""
VOLUME [ DATA_CACHE_FOLDER ]

# 3. MOUNT Model Serving Folder
VOLUME [ "/home/fedml/models_serving" ]

# 4. Enter the entrypoint
WORKDIR /home/fedml/models_serving
ENV MAIN_ENTRY=""
# if bootstrap dir is not empty, then run the bootstrap script
CMD /bin/bash ${BOOTSTRAP_DIR}; python3 ${MAIN_ENTRY}