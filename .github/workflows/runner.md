# Install  GitHub runner with your own computer:
ssh -i amir-github-actions-key.cer ubuntu@54.183.200.162
ssh -i amir-github-actions-key.cer ubuntu@52.53.164.162
ssh -i github_actions.cer ubuntu@54.153.18.24

https://github.com/FedML-AI/FedML/settings/actions/runners/new

mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.293.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.293.0/actions-runner-linux-x64-2.293.0.tar.gz
echo "06d62d551b686239a47d73e99a557d87e0e4fa62bdddcf1d74d4e6b2521f8c10  actions-runner-linux-x64-2.293.0.tar.gz" | shasum -a 256 -c
tar xzf ./actions-runner-linux-x64-2.293.0.tar.gz
./config.sh --url https://github.com/FedML-AI/FedML --token AQLQ4JAGL4KICTWDZSXVXQDCV7GSO
nohup bash run.sh > actions.log 2>&1 &

# Install GitHub runner in Ubuntu:
ssh -i "fedml-github-action.pem" ubuntu@ec2-54-176-61-229.us-west-1.compute.amazonaws.com
ssh -i "fedml-github-action.pem" ubuntu@ec2-54-219-186-81.us-west-1.compute.amazonaws.com

mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.295.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.295.0/actions-runner-linux-x64-2.295.0.tar.gz
echo "a80c1ab58be3cd4920ac2e51948723af33c2248b434a8a20bd9b3891ca4000b6  actions-runner-linux-x64-2.295.0.tar.gz" | shasum -a 256 -c
tar xzf ./actions-runner-linux-x64-2.295.0.tar.gz

sudo rpm -Uvh https://packages.microsoft.com/config/rhel/7/packages-microsoft-prod.rpm
sudo apt-get update && sudo apt-get install -y dotnet6
dotnet --version
./config.sh --url https://github.com/FedML-AI/FedML --token AXRYPLYQC7HPYXZ5ULI2PBLDAZOI2
nohup bash run.sh > actions.log 2>&1 &


# Install GitHub runner in Windows:
host: ec2-184-169-242-201.us-west-1.compute.amazonaws.com

mkdir actions-runner; cd actions-runner
Invoke-WebRequest -Uri https://github.com/actions/runner/releases/download/v2.295.0/actions-runner-win-x64-2.295.0.zip -OutFile actions-runner-win-x64-2.295.0.zip
if((Get-FileHash -Path actions-runner-win-x64-2.295.0.zip -Algorithm SHA256).Hash.ToUpper() -ne 'bd448c6ce36121eeb7f71c2c56025c1a05027c133b3cff9c7094c6bfbcc1314f'.ToUpper()){ throw 'Computed checksum did not match' }
Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory("$PWD/actions-runner-win-x64-2.295.0.zip", "$PWD")

./config.cmd --url https://github.com/FedML-AI/FedML --token AXRYPL3CQM6U6OMHN5KLJATDAZKJ4
./run.cmd