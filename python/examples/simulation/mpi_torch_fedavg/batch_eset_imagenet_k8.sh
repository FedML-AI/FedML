

/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 9 -host "gpu1:5,gpu2:4" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf '/home/zhtang/FedML/python/examples/simulation/mpi_torch_fedavg/config/eset.yaml' \
--worker_num 9 --gpu_util_parse 'gpu1:2,1,1,1;gpu2:1,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 2 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--tracking_runtime True \
--device_tag eset --tag debug --exp_name e_img_c100_k8_hier_sche --wandb_id e_img_c100_k8_hier_sche




/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 9 -host "gpu1:5,gpu2:4" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf '/home/zhtang/FedML/python/examples/simulation/mpi_torch_fedavg/config/eset.yaml' \
--worker_num 9 --gpu_util_parse 'gpu1:2,1,1,1;gpu2:1,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 2 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--runtime_est_mode history \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name e_img_c100_k8_hier --wandb_id e_img_c100_k8_hier





/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 9 -host "gpu1:5,gpu2:4" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf '/home/zhtang/FedML/python/examples/simulation/mpi_torch_fedavg/config/eset.yaml' \
--worker_num 9 --gpu_util_parse 'gpu1:2,1,1,1;gpu2:1,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 2 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode time_window \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name e_img_c100_k8_hier_sche_time --wandb_id e_img_c100_k8_hier_sche_time







/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 9 -host "gpu1:5,gpu2:4" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf '/home/zhtang/FedML/python/examples/simulation/mpi_torch_fedavg/config/eset.yaml' \
--worker_num 9 --gpu_util_parse 'gpu1:2,1,1,1;gpu2:1,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 2 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--simulation_environment_hetero cos --environment_hetero_ratio 1.0 \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name e_img_c100_k8_env_hete_hier_sche --wandb_id e_img_c100_k8_env_hete_hier_sche




/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 9 -host "gpu1:5,gpu2:4" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf '/home/zhtang/FedML/python/examples/simulation/mpi_torch_fedavg/config/eset.yaml' \
--worker_num 9 --gpu_util_parse 'gpu1:2,1,1,1;gpu2:1,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 2 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--runtime_est_mode history \
--simulation_environment_hetero cos --environment_hetero_ratio 1.0 \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name e_img_c100_k8_env_hete_hier --wandb_id e_img_c100_k8_env_hete_hier





/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 9 -host "gpu1:5,gpu2:4" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf '/home/zhtang/FedML/python/examples/simulation/mpi_torch_fedavg/config/eset.yaml' \
--worker_num 9 --gpu_util_parse 'gpu1:2,1,1,1;gpu2:1,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 2 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode time_window \
--simulation_environment_hetero cos --environment_hetero_ratio 1.0 \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name e_img_c100_k8_env_hete_hier_sche_time --wandb_id e_img_c100_k8_env_hete_hier_sche_time








/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 9 -host "gpu1:5,gpu2:4" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf '/home/zhtang/FedML/python/examples/simulation/mpi_torch_fedavg/config/eset.yaml' \
--worker_num 9 --gpu_util_parse 'gpu1:2,1,1,1;gpu2:1,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 2 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--simulation_gpu_hetero ratio --gpu_hetero_ratio 1.0 \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name e_img_c100_k8_gpu_hete_hier_sche --wandb_id e_img_c100_k8_gpu_hete_hier_sche





/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 9 -host "gpu1:5,gpu2:4" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf '/home/zhtang/FedML/python/examples/simulation/mpi_torch_fedavg/config/eset.yaml' \
--worker_num 9 --gpu_util_parse 'gpu1:2,1,1,1;gpu2:1,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 2 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--runtime_est_mode history \
--simulation_gpu_hetero ratio --gpu_hetero_ratio 1.0 \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name e_img_c100_k8_gpu_hete_hier --wandb_id e_img_c100_k8_gpu_hete_hier







/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 9 -host "gpu1:5,gpu2:4" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf '/home/zhtang/FedML/python/examples/simulation/mpi_torch_fedavg/config/eset.yaml' \
--worker_num 9 --gpu_util_parse 'gpu1:2,1,1,1;gpu2:1,1,1,1' \
--model resnet50  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset ILSVRC2012_hdf5  --data_cache_dir "/datasets/imagenet_hdf5/imagenet-shuffled.hdf5" \
--net_dataidx_map_file "/home/zhtang/FedML/python/fedml/data/ImageNet/imagenet_hdf5_c10000_n1000_alpha0.1.pth" \
--client_num_in_total 10000 --client_num_per_round 100 --comm_round 200 --epochs 2 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode time_window \
--simulation_gpu_hetero ratio --gpu_hetero_ratio 1.0 \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name e_img_c100_k8_gpu_hete_hier_sche_time --wandb_id e_img_c100_k8_gpu_hete_hier_sche_time



