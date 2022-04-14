model=$1
data=$2


python3 main_fedavg_cross_silo.py --model $model \
--dataset $data \
--partition_method homo \
--client_num_in_total 2 \
--client_num_per_round 2 \
--comm_round 2 \
--epochs 1 \
--client_optimizer sgd \
--batch_size 8 \
--lr 0.01 \
--backend MQTT_S3 \
--ci 0 \
--silo_node_rank 0 \
--nproc_per_node 1 \
--silo_rank 0 \
--pg_master_address 127.0.0.1 \
--pg_master_port 29500 \
--run_id 309 \
--data_dir ./../../../data/$data \
--mqtt_config_path ./mqtt_config.yaml \
--s3_config_path ./s3_config.yaml \
--trpc_master_config_path ./trpc_master_config.csv \
--client_ids [100,200]