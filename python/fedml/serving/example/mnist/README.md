Device Login
```sh
#Master
fedml model device login $usr_id -p -m
#Slave
fedml model device login $usr_id -p
```

## Option 1: Using UI to deploy your model
Build Package
```sh
sh build_package.sh
```
Upload to https://open.fedml.ai/serving/platform/main , -> Model Card
## Option 2: Using One-Line Command to deploy your model
```sh
export ACCOUNT_ID=YOUR_ACCOUNT_ID
export API_KEY=YOUR_API_KEY
export MASTER_DEVICE_ID=YOUR_MASTER_DEVICE_ID
export WORKER_DEVICE_ID=YOUR_WORKER_DEVICE_ID

sh one_line_deploy.sh
```