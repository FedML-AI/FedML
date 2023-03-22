
mpirun -np 5 \
-host "localhost:5" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/10workers.yaml \
--worker_num 4 --gpu_util_parse "localhost:0,2,1,1,1,0,0,0" \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedOpt_seq  --learning_rate 0.3 \
--server_optimizer sgd --server_momentum 0.9 --server_lr 1.0


mpirun -np 5 \
-host "localhost:5" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/10workers.yaml \
--worker_num 4 --gpu_util_parse "localhost:0,2,1,1,1,0,0,0" \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedOpt_seq  --learning_rate 0.1 \
--server_optimizer sgd --server_momentum 0.9 --server_lr 1.0



mpirun -np 5 \
-host "localhost:5" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/10workers.yaml \
--worker_num 4 --gpu_util_parse "localhost:0,2,1,1,1,0,0,0" \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedOpt_seq  --learning_rate 0.3 \
--server_optimizer sgd --server_momentum 0.0 --server_lr 1.0


mpirun -np 5 \
-host "localhost:5" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf config/10workers.yaml \
--worker_num 4 --gpu_util_parse "localhost:0,2,1,1,1,0,0,0" \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedOpt_seq  --learning_rate 0.1 \
--server_optimizer sgd --server_momentum 0.0 --server_lr 1.0







