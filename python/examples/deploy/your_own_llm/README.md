## Create Model
```sh
cd FedML/python/fedml/serving/example/your_own_llm/
fedml model create --name llm --config_file llm.yaml
```
## Option 1: Deploy locally
```sh
fedml model deploy --name llm --local
#INFO:     Uvicorn running on http://0.0.0.0:2345 (Press CTRL+C to quit)
curl -XPOST localhost:2345/predict -d '{"text": "Hello"}'
```
## Option 2: Deploy to the Cloud (Using TensorOpera®launch platform)
Uncomment the following line in llm.yaml,
for infomation about the configuration, please refer to TensorOpera®launch.
```yaml
# computing:
#   minimum_num_gpus: 1
#   maximum_cost_per_hour: $1
#   resource_type: A100-80G
```
Create a new model cards with serveless configuration
```sh
fedml model create --name llm_serverless --config_file llm.yaml
```
Deploy
```sh
fedml model deploy --name llm_serverless
```
## Option 3: On-premsie Deploy
Device Login
```sh
# On Master or Worker Devices
fedml login $Your_UserId_or_ApiKey
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