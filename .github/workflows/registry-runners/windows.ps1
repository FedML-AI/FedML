
$REPO = "Qigemingziba/FedML"
$ACCESS_TOKEN  = "AGMK3PY3QDYUXXXEB5LWI4DGOQIFW"
$WORKPLACE=$PWD

Set-Location actions-runner-python38
& conda activate python38
./config.cmd --url https://github.com/$REPO --name windows-python38 --token $ACCESS_TOKEN --labels self-hosted,Windows,X64,python3.8
Start-Process run.cmd start -WindowStyle Hidden

Set-Location $WORKPLACE

Set-Location actions-runner-python39
& conda activate python39
./config.cmd --url https://github.com/$REPO --name windows-python39 --token $ACCESS_TOKEN --labels self-hosted,Windows,X64,python3.9
Start-Process run.cmd start -WindowStyle Hidden

Set-Location $WORKPLACE

Set-Location actions-runner-python310
& conda activate python310
./config.cmd --url https://github.com/$REPO --name windows-python310 --token $ACCESS_TOKEN --labels self-hosted,Windows,X64,python3.10
Start-Process run.cmd start -WindowStyle Hidden

Set-Location $WORKPLACE

Set-Location actions-runner-python311
& conda activate python311
./config.cmd --url https://github.com/$REPO --name windows-python311 --token $ACCESS_TOKEN --labels self-hosted,Windows,X64,python3.11
Start-Process run.cmd start -WindowStyle Hidden

Set-Location $WORKPLACE