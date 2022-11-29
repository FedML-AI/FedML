FROM nvcr.io/nvidia/l4t-ml:r35.1.0-py3

#RUN apt-get --allow-downgrades update
#RUN apt-get install -y python3 python3-pip
#RUN ln -nsf /usr/bin/python3 /usr/bin/python
#RUN ln -nsf /usr/bin/pip3 /usr/bin/pip

#RUN sudo apt-get install python3-h5py
#RUN sudo apt install g++
#RUN sudo apt-get install python-dev
#RUN sudo apt install build-essential
#RUN python -m pip install -U pip
#RUN pip3 install --upgrade setuptools
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
RUN pip3 install h5py==2.10.0

##############################################################################
# Add docker location file
##############################################################################
ADD ./nvidia_jetson/docker-location.yml /root/fedml-client/fedml/data/docker-location.yml
ADD ./nvidia_jetson/docker-location.yml /root/fedml-server/fedml/data/docker-location.yml

RUN ln -nsf /usr/bin/python3 /usr/bin/python
RUN ln -nsf /usr/bin/pip3 /usr/bin/pip
RUN pip install --upgrade pip && pip install -U fedml
