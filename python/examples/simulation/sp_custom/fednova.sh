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



