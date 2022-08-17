# mpirun -np 3 \
# -host "localhost:3" \
# /home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp.yaml' \
# --worker_num 2 --gpu_util_parse 'localhost:0,0,0,0,0,1,1,1' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedNova  --learning_rate 0.1


mpirun -np 3 \
-host "localhost:3" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp.yaml' \
--worker_num 2 --gpu_util_parse 'localhost:0,0,0,0,0,1,1,1' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedNova  --learning_rate 0.03


mpirun -np 3 \
-host "localhost:3" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp.yaml' \
--worker_num 2 --gpu_util_parse 'localhost:0,0,0,0,0,1,1,1' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedNova  --learning_rate 0.3



# mpirun -np 3 \
# -host "localhost:3" \
# /home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp.yaml' \
# --worker_num 2 --gpu_util_parse 'localhost:0,0,0,0,0,1,1,1' \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedNova  --learning_rate 0.1




# mpirun -np 3 \
# -host "localhost:3" \
# /home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp.yaml' \
# --worker_num 2 --gpu_util_parse 'localhost:0,0,0,0,0,1,1,1' \
# --model resnet18_cifar  --group_norm_channels 0 \
# --federated_optimizer FedNova  --learning_rate 0.1






