#Create Model Card
model_name=fedllm
fedml model delete -n $model_name
fedml model create -n $model_name
#Add Files
SOURCE=./src
fedml model add -n $model_name -p $SOURCE
#Build Package
fedml model package -n $model_name
#Push Model to MLOps
echo "Your accout id: $ACCOUNT_ID"
echo "Your api key: $API_KEY"
fedml model push -n $model_name -u $ACCOUNT_ID -k $API_KEY