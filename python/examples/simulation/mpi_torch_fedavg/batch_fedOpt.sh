
mpirun -np 11 \
-host "localhost:11" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/10workers.yaml \
--worker_num 10 --gpu_util_parse "localhost:0,2,2,2,2,1,1,1" \
--model resnet18_cifar  --group_norm_channels 0 \
--federated_optimizer FedOpt  --learning_rate 0.3


mpirun -np 11 \
-host "localhost:11" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/10workers.yaml \
--worker_num 10 --gpu_util_parse "localhost:0,2,2,2,2,1,1,1" \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedOpt  --learning_rate 0.1 


# mpirun -np 9 \
# -host "localhost:9" \
# /home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/10workers.yaml \
# --model resnet18_cifar  --group_norm_channels 0 \
# --federated_optimizer FedOpt  --learning_rate 0.3


# mpirun -np 9 \
# -host "localhost:9" \
# /home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/10workers.yaml \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedOpt  --learning_rate 0.3

