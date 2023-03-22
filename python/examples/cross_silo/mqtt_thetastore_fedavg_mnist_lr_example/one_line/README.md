## Training Script

At the client side, the client ID (a.k.a rank) starts from 1.
Please also modify config/fedml_config.yaml, changing the `worker_num` the as the number of clients you plan to run.

If you want to use your customized MQTT or theta edge store server as training backends, you should uncomment and set the following lines.
#customized_training_mqtt_config: {'BROKER_HOST': 'your mqtt server address or domain name', 'MQTT_PWD': 'your mqtt password', 'BROKER_PORT': 1883, 'MQTT_KEEPALIVE': 180, 'MQTT_USER': 'your mqtt user'}
#customized_training_thetastore_config: {'store_home_dir': '~/edge-store-playground', 'upload_uri': 'http://localhost:19888/rpc', 'download_uri': 'http://localhost:19888/rpc'}

Use Theta EdgeStore as federated learning distributed storage for reading and writing models.
You should setup theta edgestore based on https://docs.thetatoken.org/docs/theta-edge-store-setup and set config parameters:
store_home_dir, upload_uri, download_uri.
If you want to use secret key to encrypt models, you should set secret key by calling Context().add("ipfs_secret_key", "your secret key")

At the server side, run the following script:
```
bash run_server.sh your_run_id
```

For client 1, run the following script:
```
bash run_client.sh 1 your_run_id
```
For client 2, run the following script:
```
bash run_client.sh 2 your_run_id
```
Note: 
1. please change the run_id in run_client/run_server scripts to your own.
2. For Windows users, please use *.bat scripts.

## A Better User-experience with FedML MLOps (open.fedml.ai)
To reduce the difficulty and complexity of these CLI commands. We recommend you to use our MLOps (open.fedml.ai).
FedML MLOps provides:
- Install Client Agent and Login
- Inviting Collaborators and group management
- Project Management
- Experiment Tracking (visualizing training results)
- monitoring device status
- visualizing system performance (including profiling flow chart)
- distributed logging
- model serving