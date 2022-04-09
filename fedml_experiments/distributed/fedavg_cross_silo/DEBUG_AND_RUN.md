## Run server on the AWS EC2 instance
1. ssh -i fedmobile.pem ubuntu@mobileapi.fedml.ai 
2. conda activate FedML 
3. cd /home/ubuntu/FLServer_Agent 
4. ./run.sh

## Debug server program on the local machine
1. Right click 'main_fedavg_cross_silo.py' in the IDEA tool
2. Click 'Debug main_fedavg_cross_silo.py' in the IDEA tool
3. Edit configuration for 'main_fedavg_cross_silo.py'
4. Add parameters: 
--model resnet56 --dataset cifar10 --partition_method homo --client_silo_num_in_total 2 --silo_num_per_round 2 --comm_round 2 --epochs 1 --client_optimizer adam --batch_size 64 --lr 0.001 --backend MQTT_S3 --ci 0 --silo_node_rank 0 --nproc_per_node 1 --silo_rank 0 --pg_master_address 127.0.0.1 --pg_master_port 29500 --run_id 169 --data_dir ./../../../data/cifar10 --mqtt_config_path ./mqtt_config.yaml --s3_config_path ./s3_config.yaml --trpc_master_config_path ./trpc_master_config.csv --client_ids [27,26]
Maybe you should modify client_silo_num_in_total, silo_num_per_round, and client_ids parameters based on your real use case
5. Click again 'Debug 'main_fedavg_cross_silo.py' in the IDEA tool
6. Then you may set break points in the python source file
7. Use the MQTT tool send the mesage with the following format:
   topic:  flserver_agent/edge_id/start_train
   content:
    {"groupid": "38", "clientLearningRate": 0.001, "partitionMethod": "homo", "starttime": 1646068794775, "trainBatchSize": 64, "edgeids": [27,26], "token": "eyJhbGciOiJIUzI1NiJ9.eyJpZCI6NjAsImFjY291bnQiOiJhbGV4LmxpYW5nIiwibG9naW5UaW1lIjoiMTY0NjA2NTY5MDAwNSIsImV4cCI6MH0.0OTXuMTfxqf2duhkBG1CQDj1UVgconnoSH0PASAEzM4", "modelName": "resnet56", "urls": ["https://fedmls3.s3.amazonaws.com/025c28be-b464-457a-ab17-851ae60767a9"], "clientOptimizer": "adam", "userids": ["60"], "clientNumPerRound": 3, "name": "1646068810", "commRound": 3, "localEpoch": 1, "runId": 169, "id": 169, "projectid": "56", "dataset": "cifar10", "communicationBackend": "MQTT_S3", "timestamp": "1646068794778"}

   Maybe you should modify edge_id, edgeids, modelName, dataset, userids parameter based on your real use case

   



