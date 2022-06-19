FROM nvcr.io/nvidia/l4t-ml:r32.6.1-py3

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

RUN ln -nsf /usr/bin/python3 /usr/bin/python
RUN ln -nsf /usr/bin/pip3 /usr/bin/pip
RUN pip3 install fedml==0.7.95