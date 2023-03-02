# Installing FedML

FedML supports Linux, MacOS, Windows, and Android.

## FedML Source Code Repository
[https://github.com/FedML-AI/FedML](https://github.com/FedML-AI/FedML)


## Installing with pip

```
pip install fedml
```

The default machine learning engine is `PyTorch`. FedML also supports `TensorFlow`, `Jax`, and `MXNet`.
You can install related engine as follows:
```
pip install "fedml[MPI]"
pip install "fedml[tensorflow]"
pip install "fedml[jax]"
pip install "fedml[mxnet]"
```
For MPI installation, it's used for local distributed simulation with MPI (https://mpi4py.readthedocs.io/en/stable/). On MacOS, the installation commands in conda environment is:
```
conda install mpi4py openmpi
```
About OpenMPI library installation for MPI, the reference is as follows: [https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html](https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html,)
For OpenMPI on MacOS, please review the following links: 
[https://betterprogramming.pub/integrating-open-mpi-with-clion-on-apple-m1-76b7815c27f2](https://formulae.brew.sh/formula/open-mpi)
[https://formulae.brew.sh/formula/open-mpi](https://formulae.brew.sh/formula/open-mpi)

The above commands work properly in Linux environment. 
For Windows/Mac OS (Intel)/Mac OS (M1), you may need to follow TensorFlow/Jax/MXNet official guidance to fix related installation issues.

## Installing FedML with Anaconda

```
conda create --name fedml python
conda activate fedml
conda install --name fedml pip
pip install fedml
```
(Note: please use python 3.7 if you met any compatability issues. We will support 3.8, 3.9, 3.10 systematically in the next iteration.)

After installation, please use "pip list | grep fedml" to check whether `fedml` is installed.


## Installing FedML from Debugging and Editable Mode
```
cd python
pip install -e ./
```

## Installing FedML from Source
```
git clone https://github.com/FedML-AI/FedML.git && \
cd ./FedML/python && \
python setup.py install
```

If you want to run examples with tensforflow, jax or mxnet, you need to install optional dependencies:
```
git clone https://github.com/FedML-AI/FedML.git && \
cd ./FedML/python && \
python install -e '.[tensorflow]'
python install -e '.[jax]'
python install -e '.[mxnet]'
```
(Notes: Tensorflow example located in tf_mqtt_s3_fedavg_mnist_lr_example directory, Jax example location in jax_haiku_mqtt_s3_fedavg_mnist_lr_example directory)

If you need to install from a specific commit (normally used for the debugging/development phase), please follow commands below:
```
git clone https://github.com/FedML-AI/FedML.git && \
cd ./FedML && git checkout e798061d62560b03e049d514e7cc8f1a753fde6b && \
cd python && \
python setup.py install
```
Please change the above commit id to your own (you can find it at [https://github.com/FedML-AI/FedML/commits/master](https://github.com/FedML-AI/FedML/commits/master))


## Running FedML in Docker (Recommended)
FedML Docker Hub: [https://hub.docker.com/repository/docker/fedml/fedml](https://hub.docker.com/repository/docker/fedml/fedml)

We recommend using FedML in the Docker environment as it circumvents complex and tedious installation debugging. Currently, we maintain docker images for two settings:

- For Linux servers with x86_64 architecture

Please refer to the following commands and remember to change `WORKSPACE` to your own.

**(1) Pull the Docker image and prepare the docker environment**
```
FEDML_DOCKER_IMAGE=fedml/fedml:latest-torch1.12.1-cuda11.3-cudnn8-devel
docker pull FEDML_DOCKER_IMAGE

# if you want to use GPUs in your host OS, please follow this link: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
sudo chmod 777 /var/run/docker.sock
```

**(2) Run Docker with interactive mode**

```
FEDML_DOCKER_IMAGE=fedml/fedml:latest-torch1.12.1-cuda11.3-cudnn8-devel
WORKSPACE=/home/chaoyanghe/sourcecode/FedML_startup/FedML

docker run -t -i -v $WORKSPACE:$WORKSPACE --shm-size=64g --ulimit nofile=65535 --ulimit memlock=-1 --privileged \
--env FEDML_NODE_INDEX=0 \
--env WORKSPACE=$WORKSPACE \
--env FEDML_NUM_NODES=1 \
--env FEDML_MAIN_NODE_INDEX=0 \
--env FEDML_RUN_ID=0 \
--env FEDML_MAIN_NODE_PRIVATE_IPV4_ADDRESS=127.0.0.1 \
--gpus all \
-u fedml --net=host \
$FEDML_DOCKER_IMAGE \
/bin/bash
```

You should now see a prompt that looks something like:
```
fedml@ChaoyangHe-GPU-RTX2080Tix4:/$ 
fedml@ChaoyangHe-GPU-RTX2080Tix4:/$ cd $WORKSPACE
fedml@ChaoyangHe-GPU-RTX2080Tix4:/home/chaoyanghe/sourcecode/FedML_startup/FedML$
```
If you want to debug in Docker container, please follow these commands
```
cd python
# You need sudo permission to install your debugging package in editable mode 
(-e means link the package to the original location, basically meaning any changes to the original package would reflect directly in your environment)
sudo pip install -e ./
```

**(3) Run Docker with multiple commands to launch your project immediately**

Here is an example of running federated learning with MNIST dataset and Logistic Regression model.
```
WORKSPACE=/home/chaoyanghe/sourcecode/FedML_startup/FedML

docker run -t -i -v $WORKSPACE:$WORKSPACE --shm-size=64g --ulimit nofile=65535 --ulimit memlock=-1 --privileged \
--env FEDML_NODE_INDEX=0 \
--env WORKSPACE=$WORKSPACE \
--env FEDML_NUM_NODES=1 \
--env FEDML_MAIN_NODE_INDEX=0 \
--env FEDML_RUN_ID=0 \
--env FEDML_MAIN_NODE_PRIVATE_IPV4_ADDRESS=127.0.0.1 \
-u fedml --net=host \
--gpus all \
$FEDML_DOCKER_IMAGE \
/bin/bash -c `cd $WORKSPACE/python/examples/simulation/mpi_torch_fedavg_mnist_lr_example; sh run_one_line_example.sh`
```

**(4) Run Docker with bootstrap.sh and entry.sh**

For advanced usage, you may need to install additional python packages or set some additional environments for your project.
In this case, we recommend that you specify the `bootstrap.sh` for the additional package installation and environment settings, and
`entry.sh`, where you launch your main program. Here is an example of running the same task in (3).

-------boostrap.sh----------
```
#!/bin/bash
echo "This is bootstrap script. You can use it to customize your additional installation and set some environment variables"

# here we upgrade fedml to the latest version.
pip install --upgrade fedml
```
-------entry.sh----------
```
#!/bin/bash
echo "This is entry script where you launch your main program."

cd $WORKSPACE/python/examples/simulation/mpi_torch_fedavg_mnist_lr_example
sh run_one_line_example.sh

```

```
WORKSPACE=/home/chaoyanghe/sourcecode/FedML_startup/FedML

docker run -t -i -v $WORKSPACE:$WORKSPACE --shm-size=64g --ulimit nofile=65535 --ulimit memlock=-1 --privileged \
--env FEDML_NODE_INDEX=0 \
--env WORKSPACE=$WORKSPACE \
--env FEDML_NUM_NODES=1 \
--env FEDML_MAIN_NODE_INDEX=0 \
--env FEDML_RUN_ID=0 \
--env FEDML_MAIN_NODE_PRIVATE_IPV4_ADDRESS=127.0.0.1 \
--env FEDML_BATCH_BOOTSTRAP=$WORKSPACE/python/scripts/docker/bootstrap.sh \
--env FEDML_BATCH_ENTRY_SCRIPT=$WORKSPACE/python/scripts/docker/entry.sh \
--gpus all \
-u fedml --net=host \
$FEDML_DOCKER_IMAGE
```

**(5) Run the interpreter in PyCharm or Visual Studio using Docker environment**

- PyCharm

[https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html#summary](https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html#summary)

- Visual Studio

[https://code.visualstudio.com/docs/remote/containers](https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html#summary)

(6) Other useful commands
```
# docker rm $(docker ps -aq)
docker container kill $(docker ps -q)
```


## Guidance for Windows Users

Please follow instructions at [Windows Installation](./install/windows.md)

## Guidance for Raspberry Pi Users 

Please follow instructions at [Raspberry Pi Installation](./install/rpi.md)

## Guidance for NVIDIA Jetson Devices

Please follow instructions at [NVIDIA Jetson Device Installation](./install/jetson.md)

## Testing if the installation succeeded
If the installation is successful, you will not see any issue when run `import fedml`.
```shell
(mnn37) chaoyanghe@Chaoyangs-MBP FedML-refactor % python
Python 3.7.7 (default, Mar 26 2020, 10:32:53) 
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import fedml
>>> 

```

## Installing FedML Android SDK/APP
Please follow the instructions at `https://github.com/FedML-AI/FedML/java/README.md`

## Troubleshooting
If you met any issues during installation or have additional installation requirements, please post issues at [https://github.com/FedML-AI/FedML/issues](https://github.com/FedML-AI/FedML/issues)