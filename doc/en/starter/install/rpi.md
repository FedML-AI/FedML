# FedML Installation on Raspberry Pi

## Run FedML with Docker (Recommended)
- Pull FedML RPI docker image
```
docker pull fedml/fedml:raspberrypi4-64-py37
```

- Run Docker with "fedml login"
```
docker run -t -i fedml/fedml:raspberrypi4-64-py37 /bin/bash

root@8bc0de2ce0e0:/usr/src/app# fedml login 299

```


Note please change the value of $YOUR_FEDML_ACCOUNT_ID to your own.

## Install Docker on Your Raspberry Pi (skip this if you already installed Docker)
1. Update and upgrade your apt-get package tool

```
sudo apt-get update && sudo apt-get upgrade
```

2. Download Docker installation script
```
curl -fsSL https://get.docker.com -o get-docker.sh
```

3. Execute the installation script
```
sudo sh get-docker.sh
```

4. Add a non-root user to the Docker group
```
sudo usermod -aG docker [your-user]
```

## Install with pip

```
pip install fedml
source ~/.profile  ## run this command if '/home/user/.local/bin' is not on PATH' after installation
```
