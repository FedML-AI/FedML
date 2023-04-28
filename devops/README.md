# FedML DevOps

We use the Jenkins to run our internal devops pipelines. So, all pipeline definition is located in the file named Jenkinsfile. 
When the developer pushes or merges codes into the dev or master branch, the Jenkins pipelines will be triggered by the webhook.
It will automatically build the related Docker Images, push the Docker Images into the AWS ECR or DockerHub, and rollout the Docker Images into the K8S cluster. 

The devops includes three parts:
1. dockerfile: 
it includes all Dockerfile for building fedml training client, server and model serving components.

2. k8s: 
it includes the deployment file or Helm Charts for deploying fedml training client, server and model serving components.

3. scripts:
it includes related scripts for building the Docker Images which will be used in the Dockerfile.
