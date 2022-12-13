
# run 10 workers, not using sequential
# mpirun -np 6 \
# -host "localhost:6" \
# /home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf config/5workers.yaml \
# --override_cmd_args






#     FedAvg
# # ===========================================================================
# mpirun -np 6 \
# -host "localhost:6" \
# /home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
# --worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedAvg  --learning_rate 0.01 --batch_size 128 \
# --dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 1000

# mpirun -np 6 \
# -host "localhost:6" \
# /home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
# --worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedAvg  --learning_rate 0.03 --batch_size 128 \
# --dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 1000




# #     FedProx
# # ===========================================================================
# mpirun -np 6 \
# -host "localhost:6" \
# /home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
# --worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedProx  --learning_rate 0.01 --batch_size 128 \
# --dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 1000

# mpirun -np 6 \
# -host "localhost:6" \
# /home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
# --worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedProx  --learning_rate 0.03 --batch_size 128 \
# --dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 1000




#     SCAFFOLD
# # ===========================================================================
# mpirun -np 6 \
# -host "localhost:6" \
# /home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
# --worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer SCAFFOLD  --learning_rate 0.01 --batch_size 128 \
# --dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 1000

# mpirun -np 6 \
# -host "localhost:6" \
# /home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
# --worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer SCAFFOLD  --learning_rate 0.03 --batch_size 128 \
# --dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 1000

mpirun -np 6 \
-host "localhost:6" \
/home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
--worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer SCAFFOLD  --learning_rate 0.1 --batch_size 128 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500

mpirun -np 6 \
-host "localhost:6" \
/home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
--worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer SCAFFOLD  --learning_rate 0.3 --batch_size 128 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500


#     FedNova
# ===========================================================================
# mpirun -np 6 \
# -host "localhost:6" \
# /home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
# --worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedNova  --learning_rate 0.01 --batch_size 128 \
# --dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 1000


# mpirun -np 6 \
# -host "localhost:6" \
# /home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
# --worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedNova  --learning_rate 0.03 --batch_size 128 \
# --dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 1000


# mpirun -np 6 \
# -host "localhost:6" \
# /home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
# --worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedNova  --learning_rate 0.1 --batch_size 128 \
# --dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 1000


# mpirun -np 6 \
# -host "localhost:6" \
# /home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
# --worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedNova  --learning_rate 0.3 --batch_size 128 \
# --dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 1000






# #    FedDyn
# # ===========================================================================
# mpirun -np 6 \
# -host "localhost:6" \
# /home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
# --worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedDyn  --learning_rate 0.01 --batch_size 128 --feddyn_alpha 0.01 \
# --dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 1000

# mpirun -np 6 \
# -host "localhost:6" \
# /home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
# --worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedDyn  --learning_rate 0.03 --batch_size 128 --feddyn_alpha 0.01 \
# --dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 1000




#     Mime with Momentum
# ===========================================================================
mpirun -np 6 \
-host "localhost:6" \
/home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
--worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer Mime  --mimelite True --batch_size 128  --weight_decay 0.0001 \
--client_optimizer sgd --learning_rate 0.01 --momentum 0.9 \
--server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 1000


mpirun -np 6 \
-host "localhost:6" \
/home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
--worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer Mime  --mimelite True --batch_size 128  --weight_decay 0.0001 \
--client_optimizer sgd --learning_rate 0.1 --momentum 0.9 \
--server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 1000



# mpirun -np 6 \
# -host "localhost:6" \
# /home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
# --worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer Mime  --mimelite True --batch_size 128  --weight_decay 0.0001 \
# --client_optimizer sgd --learning_rate 0.03 --momentum 0.9 \
# --server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
# --dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 1000



# mpirun -np 6 \
# -host "localhost:6" \
# /home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
# --worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer Mime  --mimelite True --batch_size 128  --weight_decay 0.0001 \
# --client_optimizer sgd --learning_rate 0.1 --momentum 0.9 \
# --server_optimizer sgd --server_lr 0.01  --server_momentum 0.9 \
# --dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 1000



# mpirun -np 6 \
# -host "localhost:6" \
# /home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
# --worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer Mime  --mimelite True --batch_size 128  --weight_decay 0.0001 \
# --client_optimizer sgd --learning_rate 0.3 --momentum 0.9 \
# --server_optimizer sgd --server_lr 0.01  --server_momentum 0.9 \
# --dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 1000



# mpirun -np 6 \
# -host "localhost:6" \
# /home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
# --worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer Mime  --mimelite True --batch_size 128  --weight_decay 0.0001 \
# --client_optimizer sgd --learning_rate 0.01 --momentum 0.9 \
# --server_optimizer sgd --server_lr 0.01  --server_momentum 0.9 \
# --dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 1000



# mpirun -np 6 \
# -host "localhost:6" \
# /home/chaoyanghe/anaconda3/envs/fedcv/bin/python main.py --cf 'config/5workers.yaml' \
# --worker_num 6 --gpu_util_parse 'localhost:2,0,0,1,1,1,1,0' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer Mime  --mimelite True --batch_size 128  --weight_decay 0.0001 \
# --client_optimizer sgd --learning_rate 0.03 --momentum 0.9 \
# --server_optimizer sgd --server_lr 0.01  --server_momentum 0.9 \
# --dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
# --client_num_in_total 10 --client_num_per_round 5 --comm_round 1000
































