## Create Model
```sh
cd FedML/python/fedml/serving/example/llm/
fedml model create --name llm --config_file llm.yaml
```

## Option 1: Deploy locally
```sh
fedml model deploy --name llm --local
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
Deploy
```sh
fedml model deploy --name llm --master_ids $master_id --worker_ids $client_id --user_id $usr_id --api_key $api_key
```
```sh
# Above parameters can also be passed by environment variables
export FEDML_USER_ID=YOUR_USER_ID
export FEDML_API_KEY=YOUR_API_KEY
export FEDML_MODEL_SERVE_MASTER_DEVICE_IDS=YOUR_MASTER_DEVICE_ID
export FEDML_MODEL_SERVE_WORKER_DEVICE_IDS=YOUR_WORKER_DEVICE_IDS

fedml model deploy --name llm
#Check the progress on fedml mlops UI
```