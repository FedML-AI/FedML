## Create Model
```sh
cd FedML/python/fedml/serving/example/llm/
fedml model create --name llm --config_file llm.yaml
```

## Option 1: Deploy locally
```sh
fedml model serve --name llm --local
```
## Option 2: On-premsie Deploy
Device Login
```sh
#Master
fedml model device login $usr_id -p -m
# Your FedML Edge ID is XXX
#Slave
fedml model device login $usr_id -p
# Your FedML Edge ID is XXX
```
```sh
export FEDML_USER_ID=YOUR_USER_ID
export FEDML_API_KEY=YOUR_API_KEY
export FEDML_MODEL_SERVE_MASTER_DEVICE_ID=YOUR_MASTER_DEVICE_ID
export FEDML_MODEL_SERVE_WORKER_DEVICE_IDS=YOUR_WORKER_DEVICE_IDS
#If more than one worker, use comma to separate, e.g. 1,2,3
```sh

fedml model serve --name llm
```