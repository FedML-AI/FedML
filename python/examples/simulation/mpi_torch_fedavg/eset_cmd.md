#     FedAvg
# ===========================================================================
mpirun -np 5 \
-host "localhost:5" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/5workers.yaml' \
--worker_num 5 --gpu_util_parse 'localhost:2,1,1,1' \
--model femnist  --group_norm_channels 32 \
--federated_optimizer FedAvg  --learning_rate 0.1 --batch_size 128 \
--dataset femnist  --data_cache_dir "/datasets/FederatedEMNIST/datasets" \
imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 50






mpirun -np 5 \
-host "localhost:5" \
/home/chaoyanghe/anaconda3/envs/fedml/bin/python main.py --cf 'config/5workers.yaml' \
--worker_num 5 --gpu_util_parse 'localhost:2,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg  --learning_rate 0.1 --batch_size 128 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 50





