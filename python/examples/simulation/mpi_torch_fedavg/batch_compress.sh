
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
# --device_tag sci --tag debug



# #  no compression, download whole model 
# mpirun -np 4 -host "localhost:4" \
# /home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fl_compression.yaml' \
# --worker_num 9 --gpu_util_parse 'localhost:2,0,2,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedDLC --local_cache True  --learning_rate 0.1 --batch_size 128 \
# --server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
# --feddlc_download_dense True \
# --dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
# --client_num_in_total 10 --client_num_per_round 10 --comm_round 300 --epochs 1 \
# --frequency_of_the_test 10 \
# --aggregate_seq True --hierarchical_agg True \
# --simulation_schedule "LinearFit-DP" --runtime_est_mode history \
# --tracking_runtime True \
# --enable_wandb True \
# --device_tag sci --tag debug


# #  no compression 
# mpirun -np 4 -host "localhost:4" \
# /home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fl_compression.yaml' \
# --worker_num 9 --gpu_util_parse 'localhost:0,2,2,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedDLC --local_cache True  --learning_rate 0.1 --batch_size 128 \
# --server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
# --dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
# --client_num_in_total 10 --client_num_per_round 10 --comm_round 300 --epochs 1 \
# --frequency_of_the_test 10 \
# --aggregate_seq True --hierarchical_agg True \
# --simulation_schedule "LinearFit-DP" --runtime_est_mode history \
# --tracking_runtime True \
# --enable_wandb True \
# --device_tag sci --tag debug


# #  client2server Quanti compressor
mpirun -np 6 -host "localhost:6" \
/home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fl_compression.yaml' \
--worker_num 9 --gpu_util_parse 'localhost:4,0,0,2' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedDLC --local_cache True  --learning_rate 0.1 --batch_size 128 \
--server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
--compression topk --compression_sparse_ratio 0.01 --compression_quantize_level 4 \
--down_compression no --down_compression_sparse_ratio 0.1 --feddlc_download_dense True \
--compression_warmup_round -1 \
--dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
--client_num_in_total 10 --client_num_per_round 10 --comm_round 300 --epochs 1 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--tracking_runtime True \
--enable_wandb True \
--device_tag sci --tag debug \
--run_name fl_compression_topk_upload


mpirun -np 6 -host "localhost:6" \
/home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fl_compression.yaml' \
--worker_num 9 --gpu_util_parse 'localhost:4,0,0,2' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedDLC --local_cache True  --learning_rate 0.1 --batch_size 128 \
--server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
--compression atomo --compression_sparse_ratio 0.01 --compression_quantize_level 4 \
--down_compression no --down_compression_sparse_ratio 0.1 --feddlc_download_dense True \
--compression_warmup_round -1 \
--dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
--client_num_in_total 10 --client_num_per_round 10 --comm_round 300 --epochs 1 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--tracking_runtime True \
--enable_wandb True \
--device_tag sci --tag debug \
--run_name fl_compression_atomo_upload

mpirun -np 6 -host "localhost:6" \
/home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fl_compression.yaml' \
--worker_num 9 --gpu_util_parse 'localhost:4,0,0,2' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedDLC --local_cache True  --learning_rate 0.1 --batch_size 128 \
--server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
--compression randomkec --compression_sparse_ratio 0.01 --compression_quantize_level 4 \
--down_compression no --down_compression_sparse_ratio 0.1 --feddlc_download_dense True \
--compression_warmup_round -1 \
--dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
--client_num_in_total 10 --client_num_per_round 10 --comm_round 300 --epochs 1 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--tracking_runtime True \
--enable_wandb True \
--device_tag sci --tag debug \
--run_name fl_compression_randomkec_upload


mpirun -np 6 -host "localhost:6" \
/home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fl_compression.yaml' \
--worker_num 9 --gpu_util_parse 'localhost:4,0,0,2' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedDLC --local_cache True  --learning_rate 0.1 --batch_size 128 \
--server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
--compression quantize --compression_sparse_ratio 0.01 --compression_quantize_level 4 \
--down_compression no --down_compression_sparse_ratio 0.1 --feddlc_download_dense True \
--compression_warmup_round -1 \
--dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
--client_num_in_total 10 --client_num_per_round 10 --comm_round 300 --epochs 1 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--tracking_runtime True \
--enable_wandb True \
--device_tag sci --tag debug \
--run_name fl_compression_quantize_upload



mpirun -np 6 -host "localhost:6" \
/home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fl_compression.yaml' \
--worker_num 9 --gpu_util_parse 'localhost:4,0,0,2' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedDLC --local_cache True  --learning_rate 0.1 --batch_size 128 \
--server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
--compression qsgd --compression_sparse_ratio 0.01 --compression_quantize_level 4 \
--down_compression no --down_compression_sparse_ratio 0.1 --feddlc_download_dense True \
--compression_warmup_round -1 \
--dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
--client_num_in_total 10 --client_num_per_round 10 --comm_round 300 --epochs 1 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--tracking_runtime True \
--enable_wandb True \
--device_tag sci --tag debug \
--run_name fl_compression_qsgd_upload







# #  client2server compression 
# mpirun -np 6 -host "localhost:4" \
# /home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fl_compression.yaml' \
# --worker_num 9 --gpu_util_parse 'localhost:2,0,4,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedDLC --local_cache True  --learning_rate 0.1 --batch_size 128 \
# --server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
# --compression eftopk --compression_sparse_ratio 0.01 \
# --down_compression no --down_compression_sparse_ratio 0.1 --feddlc_download_dense True \
# --compression_warmup_round 20 \
# --dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
# --client_num_in_total 10 --client_num_per_round 10 --comm_round 300 --epochs 1 \
# --frequency_of_the_test 10 \
# --aggregate_seq True --hierarchical_agg True \
# --simulation_schedule "LinearFit-DP" --runtime_est_mode history \
# --tracking_runtime True \
# --enable_wandb True \
# --device_tag sci --tag debug

# mpirun -np 6 -host "localhost:4" \
# /home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fl_compression.yaml' \
# --worker_num 9 --gpu_util_parse 'localhost:2,0,4,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedDLC --local_cache True  --learning_rate 0.1 --batch_size 128 \
# --server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
# --compression eftopk --compression_sparse_ratio 0.01 \
# --down_compression no --down_compression_sparse_ratio 0.1 --feddlc_download_dense True \
# --compression_warmup_round 50 \
# --dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
# --client_num_in_total 10 --client_num_per_round 10 --comm_round 300 --epochs 1 \
# --frequency_of_the_test 10 \
# --aggregate_seq True --hierarchical_agg True \
# --simulation_schedule "LinearFit-DP" --runtime_est_mode history \
# --tracking_runtime True \
# --enable_wandb True \
# --device_tag sci --tag debug


# --compression_warmup_round 20 \


# # double-link compression: Server2client and client2server
# mpirun -np 6 -host "localhost:6" \
# /home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fl_compression.yaml' \
# --worker_num 9 --gpu_util_parse 'localhost:2,0,4,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedDLC --local_cache True  --learning_rate 0.1 --batch_size 128 \
# --server_optimizer sgd --server_lr 0.3  --server_momentum 0.9 \
# --compression eftopk --compression_sparse_ratio 0.01 \
# --down_compression topk --down_compression_sparse_ratio 0.1 --feddlc_download_dense False \
# --compression_warmup_round 20 \
# --dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
# --client_num_in_total 10 --client_num_per_round 10 --comm_round 300 --epochs 1 \
# --frequency_of_the_test 10 \
# --aggregate_seq True --hierarchical_agg True \
# --simulation_schedule "LinearFit-DP" --runtime_est_mode history \
# --tracking_runtime True \
# --enable_wandb True \
# --device_tag sci --tag debug



# mpirun -np 6 -host "localhost:6" \
# /home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fl_compression.yaml' \
# --worker_num 9 --gpu_util_parse 'localhost:2,0,4,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedDLC --local_cache True  --learning_rate 0.1 --batch_size 128 \
# --server_optimizer sgd --server_lr 0.3  --server_momentum 0.9 \
# --compression eftopk --compression_sparse_ratio 0.01 \
# --down_compression topk --down_compression_sparse_ratio 0.1 --feddlc_download_dense False \
# --compression_warmup_round 50 \
# --dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
# --client_num_in_total 10 --client_num_per_round 10 --comm_round 300 --epochs 1 \
# --frequency_of_the_test 10 \
# --aggregate_seq True --hierarchical_agg True \
# --simulation_schedule "LinearFit-DP" --runtime_est_mode history \
# --tracking_runtime True \
# --enable_wandb True \
# --device_tag sci --tag debug




# #  no compression 
# mpirun -np 4 -host "localhost:4" \
# /home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fl_compression.yaml' \
# --worker_num 9 --gpu_util_parse 'localhost:0,2,2,0' \
# --model cifar10flnet  --group_norm_channels 32 \
# --federated_optimizer FedDLC --local_cache True  --learning_rate 0.1 --batch_size 128 \
# --server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
# --dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
# --client_num_in_total 10 --client_num_per_round 10 --comm_round 300 --epochs 1 \
# --frequency_of_the_test 10 \
# --aggregate_seq True --hierarchical_agg True \
# --simulation_schedule "LinearFit-DP" --runtime_est_mode history \
# --tracking_runtime True \
# --enable_wandb False \
# --device_tag sci --tag debug




# #  no compression 
# mpirun -np 6 -host "localhost:6" \
# /home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fl_compression.yaml' \
# --worker_num 5 --gpu_util_parse 'localhost:0,3,3,0' \
# --model cifar10flnet  --group_norm_channels 32 \
# --federated_optimizer FedDLC --local_cache True  --learning_rate 0.1 --batch_size 128 \
# --server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
# --dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
# --client_num_in_total 5 --client_num_per_round 5 --comm_round 300 --epochs 1 \
# --frequency_of_the_test 10 \
# --aggregate_seq False --hierarchical_agg False \
# --simulation_schedule "LinearFit-DP" --runtime_est_mode history \
# --tracking_runtime True \
# --enable_wandb False \
# --device_tag sci --tag debug




