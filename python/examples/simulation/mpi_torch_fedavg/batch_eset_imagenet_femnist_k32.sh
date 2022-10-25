

/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 33 -host "gpu7:5,gpu8:4,gpu11:4,gpu12:4,gpu13:4,gpu14:4,gpu15:4,gpu16:4" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf '/home/zhtang/FedML/python/examples/simulation/mpi_torch_fedavg/config/eset.yaml' \
--worker_num 33 --gpu_util_parse 'gpu7:2,1,1,1;gpu8:1,1,1,1;gpu11:1,1,1,1;gpu12:1,1,1,1;gpu13:1,1,1,1;gpu14:1,1,1,1;gpu15:1,1,1,1;gpu16:1,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 2 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--tracking_runtime True --enable_wandb False \
--device_tag eset --tag ready --exp_name e_img_c100_k32_hier_sche --wandb_id e_img_c100_k32_hier_sche




/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 33 -host "gpu7:5,gpu8:4,gpu11:4,gpu12:4,gpu13:4,gpu14:4,gpu15:4,gpu16:4" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf '/home/zhtang/FedML/python/examples/simulation/mpi_torch_fedavg/config/eset.yaml' \
--worker_num 33 --gpu_util_parse 'gpu7:2,1,1,1;gpu8:1,1,1,1;gpu11:1,1,1,1;gpu12:1,1,1,1;gpu13:1,1,1,1;gpu14:1,1,1,1;gpu15:1,1,1,1;gpu16:1,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 2 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--runtime_est_mode history \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name e_img_c100_k32_hier --wandb_id e_img_c100_k32_hier









/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 33 -host "gpu7:5,gpu8:4,gpu11:4,gpu12:4,gpu13:4,gpu14:4,gpu15:4,gpu16:4" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf '/home/zhtang/FedML/python/examples/simulation/mpi_torch_fedavg/config/eset.yaml' \
--worker_num 33 --gpu_util_parse 'gpu7:2,1,1,1;gpu8:1,1,1,1;gpu11:1,1,1,1;gpu12:1,1,1,1;gpu13:1,1,1,1;gpu14:1,1,1,1;gpu15:1,1,1,1;gpu16:1,1,1,1' \
--model resnet18_torch  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset femnist  --data_cache_dir "/home/zhtang/zhtang_data/datasets" \
--client_num_in_total 3400 --client_num_per_round 100 --comm_round 200 --epochs 10 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--tracking_runtime True \
--device_tag eset --tag ready --exp_name e_fem_c100_k32_hier_sche --wandb_id e_fem_c100_k32_hier_sche




/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 33 -host "gpu7:5,gpu8:4,gpu11:4,gpu12:4,gpu13:4,gpu14:4,gpu15:4,gpu16:4" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf '/home/zhtang/FedML/python/examples/simulation/mpi_torch_fedavg/config/eset.yaml' \
--worker_num 33 --gpu_util_parse 'gpu7:2,1,1,1;gpu8:1,1,1,1;gpu11:1,1,1,1;gpu12:1,1,1,1;gpu13:1,1,1,1;gpu14:1,1,1,1;gpu15:1,1,1,1;gpu16:1,1,1,1' \
--model resnet18_torch  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset femnist  --data_cache_dir "/home/zhtang/zhtang_data/datasets" \
--client_num_in_total 3400 --client_num_per_round 100 --comm_round 200 --epochs 10 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--runtime_est_mode history \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name e_fem_c100_k32_hier --wandb_id e_fem_c100_k32_hier



