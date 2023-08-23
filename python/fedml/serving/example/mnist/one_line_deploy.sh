#Create Model Card
model_name=mnist
fedml model delete -n $model_name
fedml model create -n $model_name
#Add Files
SOURCE=./src
fedml model add -n $model_name -p $SOURCE
#Build Package
fedml model package -n $model_name
#Upload to the mlops
echo "Your accout id: $ACCOUNT_ID"
echo "Your api key: $API_KEY"
fedml model push -n $model_name -u $ACCOUNT_ID -k $API_KEY
#Deploy Model
#Option 1: Use MLOps UI https://open.fedml.ai/serving/platform/main
#Option 2: Use CLI
echo "Your Master Device ID: $MASTER_DEVICE_ID"
echo "Your Slave Device ID: $WORKER_DEVICE_ID"
fedml model deploy -n $model_name -d "[$MASTER_DEVICE_ID, $WORKER_DEVICE_ID]" -u $ACCOUNT_ID -k $API_KEY