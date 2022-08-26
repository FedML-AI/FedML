
# mpirun -np 9 \
# -host "localhost:9" \
# /home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/optim_exp.yaml \
# --model resnet18_cifar  --group_norm_channels 0 \
# --federated_optimizer FedAvg_seq  --learning_rate 0.1


# mpirun -np 9 \
# -host "localhost:9" \
# /home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/optim_exp.yaml \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedAvg_seq  --learning_rate 0.1 


# mpirun -np 9 \
# -host "localhost:9" \
# /home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/optim_exp.yaml \
# --model resnet18_cifar  --group_norm_channels 0 \
# --federated_optimizer FedAvg_seq  --learning_rate 0.3


# mpirun -np 9 \
# -host "localhost:9" \
# /home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/optim_exp.yaml \
# --model resnet18_cifar  --group_norm_channels 32 \
# --federated_optimizer FedAvg_seq  --learning_rate 0.3



# mpirun -np 9 \
# -host "localhost:9" \
# /home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/optim_exp.yaml \
# --model resnet18  --group_norm_channels 0 \
# --federated_optimizer FedAvg_seq  --learning_rate 0.1


mpirun -np 9 \
-host "localhost:9" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/optim_exp.yaml \
--model resnet18  --group_norm_channels 32 \
--federated_optimizer FedAvg_seq  --learning_rate 0.1 


mpirun -np 9 \
-host "localhost:9" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/optim_exp.yaml \
--model resnet18  --group_norm_channels 0 \
--federated_optimizer FedAvg_seq  --learning_rate 0.3


mpirun -np 9 \
-host "localhost:9" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/optim_exp.yaml \
--model resnet18  --group_norm_channels 32 \
--federated_optimizer FedAvg_seq  --learning_rate 0.3





