


mpirun -np 3 \
-host "localhost:3" \
/home/comp/20481896/anaconda3/envs/py36/bin/python main.py --cf 'config/3workers.yaml' \
--worker_num 3 --gpu_util_parse 'localhost:1,1,1,0,0,0,0,0' \
--model resnet18_cifar  --group_norm_channels 32 \
--federated_optimizer FedAvg  --learning_rate 0.1 --batch_size 128 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/home/datasets/imagenet/imagenet_hdf5/imagenet-shuffled.hdf5" \
--client_num_in_total 10 --client_num_per_round 5 --comm_round 50










