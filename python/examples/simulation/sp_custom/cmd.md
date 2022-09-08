
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp_debug.yaml' \
--gpu_id 1 \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedNova  --learning_rate 0.1
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10"




/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp.yaml' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedNova  --learning_rate 0.1
--dataset cifar10  --data_cache_dir "/home/chaoyanghe/zhtang_FedML/python/fedml/data/cifar10"
