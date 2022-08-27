mpirun -np 9 \
-host "localhost:9" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/optim_exp.yaml \


mpirun -np 9 \
-host "localhost:9" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/debug.yaml \






# 

mpirun -np 9 \
-host "localhost:9" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/optim_exp.yaml \
--model resnet18_cifar  --group_norm_channels 0 \
--federated_optimizer FedAvg_seq  --learning_rate 0.1


mpirun -np 9 \
-host "localhost:9" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/optim_exp.yaml \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedAvg_seq  --learning_rate 0.1 


mpirun -np 9 \
-host "localhost:9" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/optim_exp.yaml \
--model resnet18_cifar  --group_norm_channels 0 \
--federated_optimizer FedAvg_seq  --learning_rate 0.3


mpirun -np 9 \
-host "localhost:9" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/optim_exp.yaml \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedAvg_seq  --learning_rate 0.3




# run 10 workers, not using sequential
mpirun -np 11 \
-host "localhost:11" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/10workers.yaml \



#  Scheduling, test performance.

mpirun -np 9 \
-host "localhost:9" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/schedule_LDAcifar10.yaml \

mpirun -np 9 \
-host "localhost:9" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/schedule_femnist.yaml \
--override_cmd_args





# debug fednova
mpirun -np 3 \
-host "localhost:3" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp_debug.yaml' \
--worker_num 2 --gpu_util_parse 'localhost:0,0,0,0,0,1,1,1' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedNova  --learning_rate 0.3


# debug fedOpt Seq
mpirun -np 3 \
-host "localhost:3" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/optim_exp_debug.yaml' \
--worker_num 2 --gpu_util_parse 'localhost:0,0,0,0,0,1,1,1' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedOpt_seq  --learning_rate 0.3



















