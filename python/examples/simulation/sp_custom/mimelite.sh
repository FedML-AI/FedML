
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
--gpu_id 3 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer Mime  --mimelite True --batch_size 128  --weight_decay 0.001 \
--client_optimizer sgd --learning_rate 0.1 --momentum 0.9 \
--server_optimizer sgd --server_lr 0.1  --server_momentum 0.9 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--initialize_all_clients True --cache_client_status False \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500



/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp.yaml' \
--gpu_id 3 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer Mime  --mimelite True --batch_size 128  --weight_decay 0.001 \
--client_optimizer sgd --learning_rate 0.1 --momentum 0.9 \
--server_optimizer sgd --server_lr 1.0  --server_momentum 0.9 \
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10" \
--initialize_all_clients True --cache_client_status False \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 500







