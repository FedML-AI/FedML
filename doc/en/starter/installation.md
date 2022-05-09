# Installing FedML

FedML supports Linux, MacOS, Windows, and Android.

## FedML Source Code Repository
[https://github.com/FedML-AI/FedML](https://github.com/FedML-AI/FedML)


## Install with pip

```
pip install fedml
```

## Install FedML with Anaconda

```
conda create --name fedml
conda activate fedml
conda install --name fedml pip
pip install fedml
```
After installation, please use `pip list | grep fedml` to check whether `fedml` is installed.

## Run FedML in Docker (Recommended)
We recommend to use FedML in Docker environment to make your life easier without caring complex and tedious installation debugging. Currently, we maintain docker images for two settings:

- For Linux servers with x86_64 architecture

Please refer to the following command and remember to change `WORKSPACE` to your own.

**(1) Pull the Docker image**
```
FEDML_DOCKER_IMAGE=fedml/fedml:cuda-11.6.0-devel-ubuntu20.04
docker pull FEDML_DOCKER_IMAGE
```
**(2) Run Docker with interactive mode**
```
FEDML_DOCKER_IMAGE=fedml/fedml:cuda-11.6.0-devel-ubuntu20.04
WORKSPACE=/home/chaoyanghe/sourcecode/FedML_startup/FedML

docker run -t -i -v $WORKSPACE:$WORKSPACE --shm-size=64g --ulimit nofile=65535 --ulimit memlock=-1 --privileged \
--env FEDML_NODE_INDEX=0 \
--env WORKSPACE=$WORKSPACE \
--env FEDML_NUM_NODES=1 \
--env FEDML_MAIN_NODE_INDEX=0 \
--env FEDML_RUN_ID=0 \
--env FEDML_MAIN_NODE_PRIVATE_IPV4_ADDRESS=127.0.0.1 \
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

**(3) Run Docker with multiple commands to launch your project immediately**

Here is an example to run federated learning with MNIST dataset and Logistic Regression model.
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
$FEDML_DOCKER_IMAGE \
/bin/bash -c `cd $WORKSPACE/python/examples/simulation/mpi_torch_fedavg_mnist_lr_example; sh run_one_line_example.sh`
```

**(4) Run Docker with bootstrap.sh and entry.sh**

For advanced usage, you may need to install additional python packages or set some additional environments for your project.
In this case, we recommend you to specify the `bootstrap.sh`, where the additional package installation and environment settings, and
`entry.sh`, where you launch your main program. Here is an example to run the same task in (3).

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
-u fedml --net=host \
$FEDML_DOCKER_IMAGE
```

**(5) Run the interpreter in PyCharm or Visual Studio using Docker environment**

- PyCharm
https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html#summary

- Visual Studio
https://code.visualstudio.com/docs/remote/containers

(6) Other useful commands
```
# docker rm $(docker ps -aq)
docker container kill $(docker ps -q)
```

- For IoT devices such as NVIDIA and Raspberry Pi 4, they are based on arm64v8 architecture. Please follow commands below.

```
coming soon
```

## Test if the installation succeeded
If the installation is correct, you will not see any issue when running `import fedml`.
```shell
(mnn37) chaoyanghe@Chaoyangs-MBP FedML-refactor % python
Python 3.7.7 (default, Mar 26 2020, 10:32:53) 
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import fedml
>>> 

```

## Install FedML Android SDK/APP
Please follow the instructions at `https://github.com/FedML-AI/FedML/java/README.md`

## Troubleshooting
If you met any issues during installation, or you have additional installation requirement, please post issues at [https://github.com/FedML-AI/FedML/issues](https://github.com/FedML-AI/FedML/issues)