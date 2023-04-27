# FedML MLOps Training client and server with Kubernetes

This tutorial will guide you to deploy your fedml client and server to target Kubernetes pods running on GPU/CPU physical nodes.

The entire workflow is as follows:
1. In the file fedml-edge-client-server/deployment-client.yml, modify the variable ACCOUNT_ID to your desired value
2. Deploy the fedml client:  ```kubectl apply -f deployment-client.yml```
3. In the file fedml-edge-client-server/deployment-server.yml, modify the variable ACCOUNT_ID to your desired value
4. Deploy the fedml server:  ```kubectl apply -f deployment-server.yml```
5. Login the FedML MLOps platform (https://open.fedml.ai), the above deployed client and server will be found in the edge devices

If you want to scale up or scal down the pods to your desired count, you may run the following command:

```kubectl scale -n $YourNameSpace --replicas=$YourDesiredPodsCount deployment/fedml-client-deployment```

```kubectl scale -n $YourNameSpace --replicas=$YourDesiredPodsCount deployment/fedml-server-deployment```

# Q&A

1. Q: How to scale up or scale down?  
A: Use the following commands: 

```kubectl scale -n $YourNameSpace --replicas=$YourDesiredPodsCount deployment/fedml-client-deployment```

```kubectl scale -n $YourNameSpace --replicas=$YourDesiredPodsCount deployment/fedml-client-deployment```

2. Q: FedML Client send online status to FedML Server via which protocol?  
A: Via MQTT


3. Q: FedML Client send model, gradient parameters to FedML Server via which protocol?  
A: Use S3 protocol to store and exchange models and use MQTT to exchange messages between FedML Client and Server


4. Q: Why do we need AWS S3?  
A: Use S3 protocol to store and exchange models.