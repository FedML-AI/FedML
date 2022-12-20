
# mpirun -np 5 -host "localhost:5" \
# /home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fl_compression.yaml' \
# --worker_num 9 --gpu_util_parse 'localhost:3,0,2,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedDLC --local_cache False  --learning_rate 0.1 --batch_size 128 \
# --server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
# --dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
# --client_num_in_total 10 --client_num_per_round 10 --comm_round 10 --epochs 1 \
# --frequency_of_the_test 5 \
# --aggregate_seq True --hierarchical_agg True \
# --simulation_schedule "LinearFit-DP" --runtime_est_mode history \
# --tracking_runtime True \
# --enable_wandb False \
# --device_tag sci --tag debug --exp_name e_fem_c100_k8_hier_sche



#  no compression, download whole model 
mpirun -np 4 -host "localhost:4" \
/home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fl_compression.yaml' \
--worker_num 9 --gpu_util_parse 'localhost:2,0,2,0' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedDLC --local_cache True  --learning_rate 0.1 --batch_size 128 \
--server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
--feddlc_download_dense True \
--dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
--client_num_in_total 10 --client_num_per_round 10 --comm_round 300 --epochs 1 \
--frequency_of_the_test 5 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--tracking_runtime True \
--enable_wandb True \
--device_tag sci --tag debug --exp_name e_fem_c100_k8_hier_sche


#  no compression 
mpirun -np 4 -host "localhost:4" \
/home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fl_compression.yaml' \
--worker_num 9 --gpu_util_parse 'localhost:2,0,2,0' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedDLC --local_cache True  --learning_rate 0.1 --batch_size 128 \
--server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
--dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
--client_num_in_total 10 --client_num_per_round 10 --comm_round 300 --epochs 1 \
--frequency_of_the_test 5 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--tracking_runtime True \
--enable_wandb True \
--device_tag sci --tag debug --exp_name e_fem_c100_k8_hier_sche



#  client2server compression 
mpirun -np 4 -host "localhost:4" \
/home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fl_compression.yaml' \
--worker_num 9 --gpu_util_parse 'localhost:2,0,2,0' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedDLC --local_cache True  --learning_rate 0.1 --batch_size 128 \
--server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
--compression eftopk --compression_sparse_ratio 0.01 \
--down_compression no --down_compression_sparse_ratio 0.1 --feddlc_download_dense True \
--dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
--client_num_in_total 10 --client_num_per_round 10 --comm_round 300 --epochs 1 \
--frequency_of_the_test 5 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--tracking_runtime True \
--enable_wandb True \
--device_tag sci --tag debug --exp_name e_fem_c100_k8_hier_sche



#  double-link compression: Server2client and client2server
mpirun -np 4 -host "localhost:4" \
/home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fl_compression.yaml' \
--worker_num 9 --gpu_util_parse 'localhost:2,0,2,0' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedDLC --local_cache True  --learning_rate 0.1 --batch_size 128 \
--server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
--compression eftopk --compression_sparse_ratio 0.01 \
--down_compression topk --down_compression_sparse_ratio 0.1 --feddlc_download_dense False \
--dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
--client_num_in_total 10 --client_num_per_round 10 --comm_round 300 --epochs 1 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--tracking_runtime True \
--enable_wandb True \
--device_tag sci --tag debug --exp_name e_fem_c100_k8_hier_sche



