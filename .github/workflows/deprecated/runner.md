# Install  GitHub runner with your own computer:
ssh -i amir-github-actions-key.cer ubuntu@54.183.200.162
ssh -i amir-github-actions-key.cer ubuntu@52.53.164.162
ssh -i github_actions.cer ubuntu@54.153.18.24

sudo rpm -Uvh https://packages.microsoft.com/config/rhel/7/packages-microsoft-prod.rpm
sudo apt-get update && sudo apt-get install -y dotnet6
dotnet --version
#install runner based on the following url: https://github.com/FedML-AI/FedML/settings/actions/runners/new?arch=x64&os=linux
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status

# Install GitHub runner in Ubuntu from AWS:
ssh -i "fedml-github-action.pem" ubuntu@ec2-54-176-61-229.us-west-1.compute.amazonaws.com
ssh -i "fedml-github-action.pem" ubuntu@ec2-54-219-186-81.us-west-1.compute.amazonaws.com
ssh -i "fedml-github-action.pem" ubuntu@ec2-54-219-187-134.us-west-1.compute.amazonaws.com

sudo rpm -Uvh https://packages.microsoft.com/config/rhel/7/packages-microsoft-prod.rpm
sudo apt-get update && sudo apt-get install -y dotnet6
dotnet --version
#install runner based on the following url: https://github.com/FedML-AI/FedML/settings/actions/runners/new?arch=x64&os=linux

sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status


# Install GitHub runner in Windows from AWS:
1. You may connect to AWS Windows server by RDP client from MAC AppStore based on the url:  https://docs.microsoft.com/en-us/windows-server/remote/remote-desktop-services/clients/remote-desktop-mac

host: ec2-184-169-242-201.us-west-1.compute.amazonaws.com

2. Enabling Windows Long Path on Windows based on the following url:
   https://www.microfocus.com/documentation/filr/filr-4/filr-desktop/t47bx2ogpfz7.html

3. install runner based on the following url: https://github.com/FedML-AI/FedML/settings/actions/runners/new?arch=x64&os=win

# Runner List
```
# Windows:
ec2-184-169-242-201.us-west-1.compute.amazonaws.com
ec2-54-193-88-223.us-west-1.compute.amazonaws.com
ec2-54-151-36-0.us-west-1.compute.amazonaws.com

# Linux:
ec2-54-176-61-229.us-west-1.compute.amazonaws.com
ec2-54-219-186-81.us-west-1.compute.amazonaws.com
ec2-54-219-187-134.us-west-1.compute.amazonaws.com
ec2-13-57-8-59.us-west-1.compute.amazonaws.com
ec2-3-101-104-5.us-west-1.compute.amazonaws.com
ec2-13-57-240-161.us-west-1.compute.amazonaws.com
ec2-3-101-61-77.us-west-1.compute.amazonaws.com

ec2-54-215-107-43.us-west-1.compute.amazonaws.com
ec2-13-56-228-205.us-west-1.compute.amazonaws.com
ec2-13-57-49-67.us-west-1.compute.amazonaws.com
ec2-18-144-32-82.us-west-1.compute.amazonaws.com
```

```
# useful commands
sudo apt update
sudo apt install libopenmpi-dev openmpi-bin
sudo apt install python3
sudo apt install python-is-python3
sudo apt install pip
pip install -U fedml
pip install mpi4py
nohup bash run.sh > action.log 2>&1 &
```