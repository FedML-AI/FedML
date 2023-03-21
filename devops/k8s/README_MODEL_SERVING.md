# FedML MLOps Model Serving with Kubernetes

This tutorial will guide you to deploy your models to target computing devices, e.g. GPU/CPU physical devices or Kubernetes pods running on GPU/CPU physical nodes.

The entire workflow is as follows:
1. create a model card by uploading your trained model file and related configuration (YAML)
2. bind (login) computing resource to FedML MLOps model serving platform (https://open.fedml.ai)
   - Kubernetes mode
   - CLI mode
3. start the deployment and get the inference API once the deployment is finished


When your model deployment is finished, you will get an endpoint URL and inference API.

```curl -XPOST https://$YourEndPointIngressDomainName/inference/api/v1/predict -H 'accept: application/json' -d'{  "model_version": "v11-Thu Jan 05 08:20:24 GMT 2023",  "model_name": "model_340_18_fedml_test_model_v11-Thu-Jan-05-08-20-24-GMT-2023",  "data": "This is our test data. Please fill in here with your real data.",  "end_point_id": 336,  "model_id": 18,  "token": "2e081ef115d04ee8adaffe5c1d0bfbac"}'```

 You may run your model deployment flow via the ModelOps(open.fedml.ai) and CLI.

Model Deployment CLI:

```
fedml model deploy -n $model_name --on_premise -d $device_id_list -u $user_id -k $user_api_key -p $deployment_extra_params

e.g. fedml model deploy -n fedml_sample_model -u 1420 -k c9356b9c4ce44363bb66366b290201 -dt md.on_premise_device -d [178077,178076]

Note: You may find your device id in the Computing Resource page at the ModelOps(open.fedml.ai) platform.
      In the $device_id_list, the master device should be the first item.
```

The above command needs two key parameters: $model_name, $device_id_list. Now let us run the following steps to prepare these parameters.

# 1. Install FedML model serving packages to computing devices (Kubernetes pods or GPU/CPU physical nodes)

## Device Login Overview
For deploying your model to Kubernetes, you should run the following steps to prepare your master device, the slave device, and the inference endpoint ingress.

The master device will be used as a model deployment and scheduling center which could forward model deployment and other messages to slave devices.

Slave devices could deploy your models into triton or other inference backend and return inference results to the master device.

Your actual inference requests will arrive at the inference endpoint ingress and then route to the inference backend located in slave devices.

Inference end point ingress will be used as your model serving endpoint URL which represents by $YourEndPointIngressDomainName.

##  Steps:
### 1). You need to specify at least one node as your master node via kubectl label CLI:

   ```kubectl label nodes <your-node-name> fedml-master-node=true```

   ```kubectl get nodes --show-labels```

### 2). You need to specify at least one node as your slave node via kubectl label CLI:

   ```kubectl label nodes <your-node-name> fedml-slave-node=true```

   ```kubectl get nodes --show-labels```

### 3). You need to specify at least one node as your inference end point ingress node via kubectl label CLI:

   ```kubectl label nodes <your-node-name> fedml-inference-ingress=true```

   ```kubectl get nodes --show-labels```

### 4). Prepare parameters will be used in the next step.
 You should fetch $YourAccountId and $YourApiKey from ModelOps(open.fedml.ai) which will be used in the next step. 

### 5). You may run the Helm Charts Installation commands to install FedML model serving packages to the above labeled nodes.

```kubectl create namespace $YourNameSpace```

```helm install --set env.fedmlAccountId="$YourAccountId" --set env.fedmlApiKey="$YourApiKey" --set env.fedmlVersion="release"  fedml-model-premise-slave fedml-model-premise-slave-latest.tgz -n $YourNameSpace```

```helm install --set env.fedmlAccountId="$YourAccountId" --set env.fedmlApiKey="$YourApiKey" --set env.fedmlVersion="release" --set "inferenceGateway.ingress.host=$YourEndPointIngressDomainName" --set "inferenceGateway.ingress.className=nginx" fedml-model-premise-master fedml-model-premise-master-latest.tgz -n $YourNameSpace```

Notes: $YourEndPointIngressDomainName is your model serving end point URL host which will be used in your inference API, e.g.

```curl -XPOST https://$YourEndPointIngressDomainName/inference/api/v1/predict -H 'accept: application/json' -d'{  "model_version": "v11-Thu Jan 05 08:20:24 GMT 2023",  "model_name": "model_340_18_fedml_test_model_v11-Thu-Jan-05-08-20-24-GMT-2023",  "data": "This is our test data. Please fill in here with your real data.",  "end_point_id": 336,  "model_id": 18,  "token": "2e081ef115d04ee8adaffe5c1d0bfbac"}'```

If you want to install FedML model serving packages to multiple pods, you should add the extra parameters to the above helm installation commands.
 
On the master device: 

  ```--set "autoscaling.enabled=true" --set replicaCount=$InstanceNumber```

On the slave device:

```--set "autoscaling.enabled=true" --set replicaCount=$InstanceNumber```

On the inference endpoint ingress device:

```--set "inferenceGateway.replicaCount=$InstanceNumber"```

If you install FedML model serving packages on GCP k8s cluster, first, you should install Nginx Ingress Controller based on the following link:

[https://github.com/kubernetes/ingress-nginx/blob/main/docs/deploy/index.md#gce-gke](https://github.com/kubernetes/ingress-nginx/blob/main/docs/deploy/index.md#gce-gke)

Moreover, on GCP k8s cluster, you should set up your GPU nodes based on the following link:

[https://cloud.google.com/kubernetes-engine/docs/how-to/gpus](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus)

After you have installed FedML model serving packages, you may run the helm upgrade commands to modify parameters.

e.g.
```helm upgrade --set "autoscaling.enabled=true" --set replicaCount=$InstanceNumber fedml-model-premise-master fedml-model-premise-master-0.7.397.tgz -n $YourNameSpace```

### 6). Config your CNAME record in your DNS provider (Godaddy, wordpress, AWS Route 53...)
#### (a). Find the Kubernetes nginx ingress named 'fedml-model-inference-gateway' in your Kubernetes cluster.
#### (b). Fetch its gateway address, e.g. a865e3a1e9aa54c71b50e3a6c764cbd3-337285825.us-west-1.elb.amazonaws.com
#### (c). Set your CNAME record, config $YourEndPointIngressDomainName to point to the above gateway address 
  

## 2. Make the model package for your trained model files.
Use the following CLI:

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

## 3. Create the model card via ModelOps or CLI
List model in the remote model repository:
```fedml model list-remote -n $model_name -u $user_id -k $user_api_key```

Build local model repository as zip model package:
```fedml model package -n $model_name```

Push local model repository to ModelOps(open.fedml.ai):
```fedml model push -n $model_name -u $user_id -k $user_api_key```

Pull remote model(ModelOps) to local model repository:
```fedml model pull -n $model_name -u $user_id -k $user_api_key```



# Q&A

1. Q: Supports automatically scale?
A: Yes. Call CLI `helm upgrade`. For example, you can do upgrade by using the following CLI:

```helm upgrade --set "autoscaling.enabled=true" --set replicaCount=$InstanceNumber fedml-model-premise-master fedml-model-premise-master-0.7.397.tgz -n $YourNameSpace```


2. Q: Does the inference endpoint supports private IP? \
A: Yes.


4. Q: During deployment, what if the k8s service does not have a public IP? \
A: During deployment, we don't need to initiate access to your k8s service from open.fedml.ai, only your k8s cluster can initiate access to open.fedml.ai