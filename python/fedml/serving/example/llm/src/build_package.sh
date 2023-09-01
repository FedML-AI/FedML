#Create Model Card
model_name=fedllm
fedml model delete -n $model_name
fedml model create -n $model_name
#Add Files
SOURCE=./src
fedml model add -n $model_name -p $SOURCE
#Build Package
fedml model package -n $model_name