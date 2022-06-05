# FedML Installation on Windows OS

This tutorial guides you to install FedML python library on Windows. The following instruction has been tested for Windows 11 on 64-bit operating system, x64-based processor. Please post issues to our GitHub if you met any issues.

- Python Installation on Windows: https://k0nze.dev/posts/how-install-python-on-windows-11/?msclkid=3785d45dce8911ecbae4ea89f38f9f29

- Conda Installation on Windows: https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html

- Git Installation on Windows: https://git-scm.com/download/win

- GitHub Installation on Windows: https://nerdschalk.com/how-to-install-and-use-git-on-windows-11/?msclkid=b6ebbab3ce8a11ecb8af70df029276d3

## Install with conda

```
conda create --name fedml python=3.7
conda activate fedml
conda install --name fedml pip
pip install fedml
```

## Install MPI on Windows
https://stackoverflow.com/questions/54615336/how-to-install-mpi4py-on-windows-10-with-msmpi

## Install from the source code
First please install "Microsoft C++ Build Tools":
https://visualstudio.microsoft.com/visual-cpp-build-tools/

## Install from Docker

https://docs.docker.com/desktop/windows/install/