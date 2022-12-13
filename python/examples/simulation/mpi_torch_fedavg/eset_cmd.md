
--mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ib0 \
    --mca btl_openib_want_fork_support 1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=ib0 \
    -x NCCL_DEBUG=VERSION \
    -x HOROVOD_CACHE_CAPACITY=0"

-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
-x HOROVOD_CACHE_CAPACITY=0" \

#     FedAvg
# ===========================================================================

mpirun -np 5 \
-host "localhost:5" \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf 'config/eset.yaml' \
--worker_num 5 --gpu_util_parse 'localhost:2,1,1,1' \
--model resnet18_torch  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset femnist  --data_cache_dir "/home/zhtang/zhtang_data/datasets" \
--client_num_in_total 3400 --client_num_per_round 100 --comm_round 200 --epochs 1 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--runtime_est_mode history \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name femnist_c100_k10_hier





mpirun -np 9 \
-host "gpu1:5,gpu3:4" \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf '/home/zhtang/FedML/python/examples/simulation/mpi_torch_fedavg/config/eset.yaml' \
--worker_num 9 --gpu_util_parse 'gpu1:2,1,1,1;gpu3:0,2,1,1' \
--model resnet18_torch  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset femnist  --data_cache_dir "/home/zhtang/zhtang_data/datasets" \
--client_num_in_total 3400 --client_num_per_round 100 --comm_round 200 --epochs 1 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--runtime_est_mode history \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name femnist_c100_k10_hier


MPIPATH=/home/esetstore/.local/openmpi-4.0.1

-bind-to none -map-by slot \

-host "gpu8:5,gpu9:4" \

-hostfile cluster8 \


/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 8 \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \

/home/esetstore/.local/openmpi-4.0.1

env MPICC=/home/esetstore/.local/openmpi-4.0.1/bin/mpicc python -m pip install mpi4py

env MPICC=/path/to/mpicc conda install mpi4py


mpirun -np 9 \
-host "gpu8:5,gpu9:4" \




/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 9 -host "gpu13:5,gpu14:4" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf '/home/zhtang/FedML/python/examples/simulation/mpi_torch_fedavg/config/eset.yaml' \
--worker_num 9 --gpu_util_parse 'gpu13:2,1,1,1;gpu14:1,1,1,1' \
--model resnet18_torch  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset femnist  --data_cache_dir "/home/zhtang/zhtang_data/datasets" \
--client_num_in_total 3400 --client_num_per_round 100 --comm_round 200 --epochs 1 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--runtime_est_mode history \
--tracking_runtime True --enable_wandb False \
--device_tag eset --tag Ready --exp_name femnist_c100_k10_hier







#     Imagenet
# ===========================================================================


mpirun -np 5 \
-host "localhost:5" \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf 'config/eset.yaml' \
--worker_num 5 --gpu_util_parse 'localhost:2,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 5 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name imagenet_c10000_k5_hier_sche


mpirun -np 5 \
-host "localhost:5" \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf 'config/eset.yaml' \
--worker_num 5 --gpu_util_parse 'localhost:2,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 5 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--runtime_est_mode history \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name imagenet_c10000_k5_hier


mpirun -np 5 \
-host "localhost:5" \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf 'config/eset.yaml' \
--worker_num 5 --gpu_util_parse 'localhost:2,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 5 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode time_window \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name imagenet_c10000_k5_hier_sche_time







/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 5 -host "localhost:5" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf 'config/eset.yaml' \
--worker_num 5 --gpu_util_parse 'localhost:2,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 5 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_environment_hetero cos --environment_hetero_ratio 1.0 \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name imagenet_c10000_k5_env_hete_hier_sche


mpirun -np 5 \
-host "localhost:5" \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf 'config/eset.yaml' \
--worker_num 5 --gpu_util_parse 'localhost:2,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 5 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_environment_hetero cos --environment_hetero_ratio 1.0 \
--runtime_est_mode history \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name imagenet_c10000_k5_env_hete_hier


mpirun -np 5 \
-host "localhost:5" \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf 'config/eset.yaml' \
--worker_num 5 --gpu_util_parse 'localhost:2,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 5 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_environment_hetero cos --environment_hetero_ratio 1.0 \
--simulation_schedule "LinearFit-DP" --runtime_est_mode time_window \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name imagenet_c10000_k5_env_hete_hier_sche_time







/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 5 -host "localhost:5" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf 'config/eset.yaml' \
--worker_num 5 --gpu_util_parse 'localhost:2,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 5 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_gpu_hetero ratio --gpu_hetero_ratio 1.0 \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name imagenet_c10000_k5_env_hete_hier_sche




/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 5 -host "localhost:5" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf 'config/eset.yaml' \
--worker_num 5 --gpu_util_parse 'localhost:2,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 5 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_gpu_hetero ratio --gpu_hetero_ratio 1.0 \
--runtime_est_mode history \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name imagenet_c10000_k5_env_hete_hier


# Not run
/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 5 -host "localhost:5" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf 'config/eset.yaml' \
--worker_num 5 --gpu_util_parse 'localhost:2,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 5 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_gpu_hetero ratio --gpu_hetero_ratio 1.0 \
--simulation_schedule "LinearFit-DP" --runtime_est_mode time_window \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name imagenet_c10000_k5_env_hete_hier_sche_time
















