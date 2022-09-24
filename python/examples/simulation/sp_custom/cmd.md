
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp_debug.yaml' \
--gpu_id 1 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedNova  --learning_rate 0.1 --batch_size 128 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500



/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp_debug.yaml' \
--gpu_id 6 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer SCAFFOLD  --learning_rate 0.1 --batch_size 128 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--initialize_all_clients False --cache_client_status False \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500 


/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp_debug.yaml' \
--gpu_id 6 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer SCAFFOLD  --learning_rate 0.1 --batch_size 128 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--initialize_all_clients True --cache_client_status False \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500 



/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp_debug.yaml' \
--gpu_id 4 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedProx  --learning_rate 0.1 --batch_size 128 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500




/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp_debug.yaml' \
--gpu_id 4 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedDyn  --learning_rate 0.1 --batch_size 128  --weight_decay 0.001 --feddyn_alpha 0.01 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--initialize_all_clients True --cache_client_status False \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500




/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp_debug.yaml' \
--gpu_id 7 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer Mime  --batch_size 128  --weight_decay 0.001 \
--client_optimizer sgd --learning_rate 0.01 --momentum 0.9 \
--server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--initialize_all_clients True --cache_client_status False \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500



/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp_debug.yaml' \
--gpu_id 3 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer Mime  --mimelite True --batch_size 128  --weight_decay 0.001 \
--client_optimizer sgd --learning_rate 0.01 --momentum 0.9 \
--server_optimizer sgd --server_lr 0.01  --server_momentum 0.9 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--initialize_all_clients True --cache_client_status False \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500



#     FedNova
# ===========================================================================
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp.yaml' \
--gpu_id 4 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedNova  --learning_rate 0.3 --batch_size 128 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500


/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp.yaml' \
--gpu_id 4 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedNova  --learning_rate 0.01 --batch_size 128 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500


/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp.yaml' \
--gpu_id 4 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedNova  --learning_rate 0.001 --batch_size 128 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500

# ===========================================================================


/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp.yaml' \
--gpu_id 5 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer SCAFFOLD  --learning_rate 0.1 --batch_size 128 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500



/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp.yaml' \
--gpu_id 6 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer SCAFFOLD  --learning_rate 0.1  --batch_size 128 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--initialize_all_clients True --cache_client_status False \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500 



/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp.yaml' \
--gpu_id 4 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedProx  --learning_rate 0.1 --batch_size 128 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500


/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp.yaml' \
--gpu_id 4 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedDyn  --learning_rate 0.1 --batch_size 128  --feddyn_alpha 0.01 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500



/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp.yaml' \
--gpu_id 3 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer Mime  --mimelite True --batch_size 128  --weight_decay 0.001 \
--client_optimizer sgd --learning_rate 0.01 --momentum 0.9 \
--server_optimizer sgd --server_lr 0.01  --server_momentum 0.9 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--initialize_all_clients True --cache_client_status False \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500




/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp.yaml' \
--gpu_id 2 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer Mime  --mimelite True --batch_size 128  --weight_decay 0.001 \
--client_optimizer sgd --learning_rate 0.1 --momentum 0.9 \
--server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--initialize_all_clients True --cache_client_status False \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500





/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp.yaml' \
--gpu_id 2 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer Mime --batch_size 128  --weight_decay 0.001 \
--client_optimizer sgd --learning_rate 0.1 --momentum 0.9 \
--server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--initialize_all_clients True --cache_client_status False \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500




/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp.yaml' \
--gpu_id 2 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer Mime --batch_size 128  --weight_decay 0.001 \
--client_optimizer sgd --learning_rate 0.01 --momentum 0.9 \
--server_optimizer sgd --server_lr 0.01  --server_momentum 0.9 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--initialize_all_clients True --cache_client_status False \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500
























