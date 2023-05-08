
ARG PYTHON_BASE_IMAGE=python:3.8
FROM ${PYTHON_BASE_IMAGE}

ENV USER fedml
ENV HOME /home/$USER

##############################################################################
# Add docker location file
##############################################################################
ADD ./light/docker-location.yml /home/fedml/fedml-client/fedml/data/docker-location.yml

RUN pip3 install fedml


