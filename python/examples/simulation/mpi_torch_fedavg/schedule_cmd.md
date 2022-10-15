mpirun -np 6 \
-host "localhost:6" \
/home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/lambda3.yaml' \
--worker_num 6 --gpu_util_parse 'localhost:0,2,1,1,0,0,1,1' \
--model resnet18_torch  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset femnist  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/FederatedEMNIST/datasets" \
--client_num_in_total 3400 --client_num_per_round 100 --comm_round 50 --epochs 10 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--simulation_gpu_hetero ratio --gpu_hetero_ratio 1.0 \
--simulation_environment_hetero cos --environment_hetero_ratio 1.0 \
--tracking_runtime True \
--device_tag lambda3 --tag debug







mpirun -np 6 \
-host "localhost:6" \
/home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/lambda3.yaml' \
--worker_num 6 --gpu_util_parse 'localhost:0,2,1,1,0,0,1,1' \
--model resnet18_torch  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset femnist  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/FederatedEMNIST/datasets" \
--client_num_in_total 3400 --client_num_per_round 100 --comm_round 50 --epochs 10 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--simulation_gpu_hetero ratio --gpu_hetero_ratio 1.0 \
--simulation_environment_hetero cos --environment_hetero_ratio 1.0 \
--tracking_runtime True \
--device_tag lambda3 --tag debug















