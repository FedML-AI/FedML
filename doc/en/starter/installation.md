# Installing FedML

FedML supports Linux, MacOS, Windows, and Android.

## FedML Source Code Repository
[https://github.com/FedML-AI/FedML](https://github.com/FedML-AI/FedML)


## Install Official Release

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
After installation, please use `pip list` to check whether `fedml` is installed.

[comment]: <> (## Launch FedML in Docker)

[comment]: <> (For users who prefer docker environment, we maintain [FedML Docker Hub]&#40;https://public.ecr.aws/x6k8q1x9/fedml-cross-silo-cpu:latest&#41;. )

[comment]: <> (Please follow the following script to install FedML with Docker Image:)

[comment]: <> (```)

[comment]: <> (docker run public.ecr.aws/x6k8q1x9/fedml-cross-silo-cpu:latest)

[comment]: <> (```)

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