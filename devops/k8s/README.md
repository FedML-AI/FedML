# Three Steps for easily using FedML ModelOps(mode.fedml.ai) platform
## 1. Make the model package for your trained model files.
Use the following commands:
Create local model repository:
```fedml model create -n $model_name```

Delete local model repository:
```fedml model delete -n $model_name -f $model_file_name```

Add file to local model repository:
```fedml model add -n $model_name -p $model_file_path```

Remove file from local model repository:
```fedml model remove -n $model_name -f $model_file_name```

List model in the local model repository:
```fedml model list -n $model_name```

Build local model repository as zip model package:
```fedml model package -n $model_name```

## 2. Create the model card via ModelOps or CLI
List model in the remote model repository:
```fedml model list-remote -n $model_name -u $user_id -k $user_api_key```

Build local model repository as zip model package:
```fedml model package -n $model_name```

Push local model repository to ModelOps(model.fedml.ai):
```fedml model push -n $model_name -u $user_id -k $user_api_key```

Pull remote model(ModelOps) to local model repository:
```fedml model pull -n $model_name -u $user_id -k $user_api_key```

## 3. Create the end point for deploying the model to specify computing devices via ModelOps or CLI
```
fedml model deploy -n $model_name -dt $device_type(options:md.on_premise_device/md.fedml_cloud_device) -d $master_device_id -u $user_id -k $user_api_key -p $deployment_extra_params
```

### Prerequisite steps: Install FedML master and slave device package to Kubernetes Clusters via Helm Charts (Automatically login to ModelOps)
#### Overview
Here we will provide guide on how to run FedML master and slave device package (Helm Charts) on your Kubernetes cluster.

The master device will be used as model deployment and scheduling center which could forward end point creation and other messages to slave devices.

Slave devices could deploy your models into triton or other inference backend and return inference results to the master device.
Your actually inference requests will arrive to our inference ingress gateway, and then route to inference backend located in slave devices.

#### CLI
```kubectl create namespace $YourNameSpace```

```helm install --set env.fedmlAccountId="$YourAccountId" --set env.fedmlApiKey="$YourApiKey" --set env.fedmlVersion="release"  fedml-model-premise-slave fedml-model-premise-slave-0.7.377.tgz -n $YourNameSpace```

```helm install --set env.fedmlAccountId="$YourAccountId" --set env.fedmlApiKey="$YourApiKey" --set env.fedmlVersion="release" --set "inferenceGateway.ingress.host=$YourIngressDomainName" fedml-model-premise-master fedml-model-premise-master-0.7.377.tgz -n $YourNameSpace```

#### Prerequisites:
1. You should get $YourAccountId and $YourApiKey from ModelOps(model.fedml.ai) and fill in here.
And set $YourIngressDomainName to your kubernetes nginx ingress address named fedml-model-inference-gateway in $YourNameSpace,
e.g. a865e3a1e9aa54c71b50e3a6c764cbd3-337285825.us-west-1.elb.amazonaws.com. 
Which should be set your CNAME record in your DNS provider, e.g. godaddy, wordpress, AWS Route 53.

2. You need to specify at least one node as your master node via kubectl label CLI:

   ```kubectl label nodes <your-node-name> fedml-master-node=true```

   ```kubectl get nodes --show-labels```

3. You need to specify at least one node as your slave node via kubectl label CLI:

   ```kubectl label nodes <your-node-name> fedml-slave-node=true```

   ```kubectl get nodes --show-labels``` 

4. You need to specify at least one node as your inference ingress node via kubectl label CLI:

   ```kubectl label nodes <your-node-name> fedml-inference-ingress=true```

   ```kubectl get nodes --show-labels```






