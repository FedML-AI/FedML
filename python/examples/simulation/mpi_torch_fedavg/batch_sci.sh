
mpirun -np 6 -host "localhost:6" \
/home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fedml_config.yaml' \
--worker_num 5 --gpu_util_parse 'localhost:6,0,0,0' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 5 --epochs 1 \
--frequency_of_the_test 3 \
--aggregate_seq False --hierarchical_agg False \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--tracking_runtime True \
--enable_wandb False \
--device_tag sci --tag debug --exp_name e_fem_c100_k8_hier_sche --wandb_id e_fem_c100_k8_hier_sche



mpirun -np 6 -host "localhost:6" \
/home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fedml_config.yaml' \
--worker_num 5 --gpu_util_parse 'localhost:6,0,0,0' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedProx --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 5 --epochs 1 \
--frequency_of_the_test 3 \
--aggregate_seq False --hierarchical_agg False \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--tracking_runtime True \
--enable_wandb False \
--device_tag sci --tag debug --exp_name e_fem_c100_k8_hier_sche --wandb_id e_fem_c100_k8_hier_sche





# mpirun -np 3 -host "localhost:3" \
# /home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fedml_config.yaml' \
# --worker_num 9 --gpu_util_parse 'localhost:3,0,0,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
# --dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 5 --epochs 1 \
# --frequency_of_the_test 3 \
# --aggregate_seq True --hierarchical_agg True \
# --simulation_schedule "LinearFit-DP" --runtime_est_mode history \
# --tracking_runtime True \
# --enable_wandb False \
# --device_tag sci --tag debug --exp_name e_fem_c100_k8_hier_sche --wandb_id e_fem_c100_k8_hier_sche



# mpirun -np 3 -host "localhost:3" \
# /home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fedml_config.yaml' \
# --worker_num 9 --gpu_util_parse 'localhost:3,0,0,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedProx --local_cache False  --learning_rate 0.05 --batch_size 20 \
# --dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 5 --epochs 1 \
# --frequency_of_the_test 3 \
# --aggregate_seq True --hierarchical_agg True \
# --simulation_schedule "LinearFit-DP" --runtime_est_mode history \
# --tracking_runtime True \
# --enable_wandb False \
# --device_tag sci --tag debug --exp_name e_fem_c100_k8_hier_sche --wandb_id e_fem_c100_k8_hier_sche







# mpirun -np 3 -host "localhost:3" \
# /home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fedml_config.yaml' \
# --worker_num 9 --gpu_util_parse 'localhost:3,0,0,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedOpt --local_cache False  --learning_rate 0.05 --batch_size 20 \
# --server_optimizer sgd --server_lr 0.03  --server_momentum 0.9 \
# --dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 5 --epochs 1 \
# --frequency_of_the_test 3 \
# --aggregate_seq True --hierarchical_agg True \
# --simulation_schedule "LinearFit-DP" --runtime_est_mode history \
# --tracking_runtime True \
# --enable_wandb False \
# --device_tag sci --tag debug --exp_name e_fem_c100_k8_hier_sche --wandb_id e_fem_c100_k8_hier_sche




# mpirun -np 3 -host "localhost:3" \
# /home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fedml_config.yaml' \
# --worker_num 9 --gpu_util_parse 'localhost:3,0,0,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedNova --local_cache False  --learning_rate 0.05 --batch_size 20 \
# --dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 5 --epochs 1 \
# --frequency_of_the_test 3 \
# --aggregate_seq True --hierarchical_agg True \
# --simulation_schedule "LinearFit-DP" --runtime_est_mode history \
# --tracking_runtime True \
# --enable_wandb False \
# --device_tag sci --tag debug --exp_name e_fem_c100_k8_hier_sche --wandb_id e_fem_c100_k8_hier_sche






# mpirun -np 3 -host "localhost:3" \
# /home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fedml_config.yaml' \
# --worker_num 9 --gpu_util_parse 'localhost:3,0,0,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer SCAFFOLD --local_cache True --learning_rate 0.01 --batch_size 20 \
# --dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 5 --epochs 1 \
# --frequency_of_the_test 3 \
# --aggregate_seq True --hierarchical_agg True \
# --simulation_schedule "LinearFit-DP" --runtime_est_mode history \
# --tracking_runtime True \
# --enable_wandb False \
# --device_tag sci --tag debug --exp_name e_fem_c100_k8_hier_sche --wandb_id e_fem_c100_k8_hier_sche






# mpirun -np 3 -host "localhost:3" \
# /home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fedml_config.yaml' \
# --worker_num 9 --gpu_util_parse 'localhost:3,0,0,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedDyn --feddyn_alpha 0.01 --local_cache True \
# --learning_rate 0.01 --batch_size 20 \
# --dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 5 --epochs 1 \
# --frequency_of_the_test 3 \
# --aggregate_seq True --hierarchical_agg True \
# --simulation_schedule "LinearFit-DP" --runtime_est_mode history \
# --tracking_runtime True \
# --enable_wandb False \
# --device_tag sci --tag debug --exp_name e_fem_c100_k8_hier_sche --wandb_id e_fem_c100_k8_hier_sche




# mpirun -np 3 -host "localhost:3" \
# /home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/fedml_config.yaml' \
# --worker_num 9 --gpu_util_parse 'localhost:3,0,0,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer Mime --mimelite True  \
# --client_optimizer sgd --learning_rate 0.05 --momentum 0.9 --batch_size 20 \
# --server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
# --dataset cifar10  --data_cache_dir "/home/comp/20481896/datasets/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 5 --epochs 1 \
# --frequency_of_the_test 3 \
# --aggregate_seq True --hierarchical_agg True \
# --simulation_schedule "LinearFit-DP" --runtime_est_mode history \
# --tracking_runtime True \
# --enable_wandb False \
# --device_tag sci --tag debug --exp_name e_fem_c100_k8_hier_sche --wandb_id e_fem_c100_k8_hier_sche






















