set REPO=Qigemingziba/FedML
set ACCESS_TOKEN=AGMK3P4W5EM5PXNYTZXXIMTGNF4MW
set WORKPLACE=%pwd%
mkdir actions-runner-python38; cd actions-runner-python38
conda activate python38
Invoke-WebRequest -Uri https://github.com/actions/runner/releases/download/v2.317.0/actions-runner-win-x64-2.317.0.zip -OutFile actions-runner-win-x64-2.317.0.zip
Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory("$PWD/actions-runner-win-x64-2.317.0.zip", "$PWD")
./config.cmd --url https://github.com/%REPO% --token %ACCESS_TOKEN% --labels self-hosted,Windows,X64,python3.8
.\run.cmd install
.\run.cmd start

cd WORKPLACE
mkdir actions-runner-python39; cd actions-runner-python39
conda activate python39
Invoke-WebRequest -Uri https://github.com/actions/runner/releases/download/v2.317.0/actions-runner-win-x64-2.317.0.zip -OutFile actions-runner-win-x64-2.317.0.zip
Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory("$PWD/actions-runner-win-x64-2.317.0.zip", "$PWD")
./config.cmd --url https://github.com/%REPO% --token %ACCESS_TOKEN% --labels self-hosted,Windows,X64,python3.9
.\run.cmd install
.\run.cmd start

cd WORKPLACE
mkdir actions-runner-python310; cd actions-runner-python310
conda activate python310
Invoke-WebRequest -Uri https://github.com/actions/runner/releases/download/v2.317.0/actions-runner-win-x64-2.317.0.zip -OutFile actions-runner-win-x64-2.317.0.zip
Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory("$PWD/actions-runner-win-x64-2.317.0.zip", "$PWD")
./config.cmd --url https://github.com/%REPO% --token %ACCESS_TOKEN% --labels self-hosted,Windows,X64,python3.10
.\run.cmd install
.\run.cmd start

cd WORKPLACE
mkdir actions-runner-python311; cd actions-runner-python311
conda activate python311
Invoke-WebRequest -Uri https://github.com/actions/runner/releases/download/v2.317.0/actions-runner-win-x64-2.317.0.zip -OutFile actions-runner-win-x64-2.317.0.zip
Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory("$PWD/actions-runner-win-x64-2.317.0.zip", "$PWD")
./config.cmd --url https://github.com/%REPO% --token %ACCESS_TOKEN% --labels self-hosted,Windows,X64,python3.11
.\run.cmd install
.\run.cmd start

