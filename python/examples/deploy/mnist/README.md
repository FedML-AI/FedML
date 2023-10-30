## Create Model
```sh
#FedML/python/fedml/serving/example/mnist/
fedml model create --name mnist --config_file mnist.yaml
```
## Option 1: Deploy locally
```sh
fedml model deploy --name mnist --local
#INFO:     Uvicorn running on http://0.0.0.0:2345 (Press CTRL+C to quit)
curl -XPOST localhost:2345/predict -d '{"arr":[$DATA]}'
#For $DATA, please check the request_input_example, it is a 28*28=784 float array
#Output:{"generated_text":"tensor([0.2333, 0.5296, 0.4350, 0.4537, 0.5424, 0.4583, 0.4803, 0.2862, 0.5507,\n        0.8683], grad_fn=<SigmoidBackward0>)"}
```
## Option 2: Deploy to the Cloud (Using fedml®launch platform)
Uncomment the following line in mnist.yaml,
for infomation about the configuration, please refer to fedml®launch.
```yaml
# computing:
#   minimum_num_gpus: 1
#   maximum_cost_per_hour: $1
#   resource_type: A100-80G 
```
Create a new model cards with serveless configuration
```sh
fedml model create --name mnist_serverless --config_file mnist.yaml
```
Deploy
```sh
fedml model deploy --name mnist_serverless
```
## Option 3: On-premsie Deploy
Device Login
```sh
# On Master or Worker Devices
fedml login $Your_UserId_or_ApiKey
```
Deploy
```sh
fedml model deploy --name mnist --master_ids $master_id --worker_ids $client_id --user_id $usr_id --api_key $api_key
```
```sh
# Above parameters can also be passed by environment variables
export FEDML_USER_ID=YOUR_USER_ID
export FEDML_API_KEY=YOUR_API_KEY
export FEDML_MODEL_SERVE_MASTER_DEVICE_IDS=YOUR_MASTER_DEVICE_ID
export FEDML_MODEL_SERVE_WORKER_DEVICE_IDS=YOUR_WORKER_DEVICE_IDS

fedml model deploy --name mnist
```