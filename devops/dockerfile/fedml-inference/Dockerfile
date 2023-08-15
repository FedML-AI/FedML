# TODO: Use a lightweight base image
ARG BASE_IMAGE=fedml/fedml:latest-torch1.13.1-cuda11.6-cudnn8-devel
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
ENTRYPOINT [ "python3" ]
CMD [ MAIN_ENTRY ]