

/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 9 -host "gpu10:5,gpu11:4" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf '/home/zhtang/FedML/python/examples/simulation/mpi_torch_fedavg/config/eset.yaml' \
--worker_num 9 --gpu_util_parse 'gpu10:2,1,1,1;gpu11:1,1,1,1' \
--model resnet18_torch  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset femnist  --data_cache_dir "/home/zhtang/zhtang_data/datasets" \
--client_num_in_total 3400 --client_num_per_round 1000 --comm_round 100 --epochs 10 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--simulation_schedule "LinearFit-DP" --runtime_est_mode history \
--tracking_runtime True \
--device_tag eset --tag debug --exp_name e_fem_c1000_k8_hier_sche --wandb_id e_fem_c1000_k8_hier_sche




/home/esetstore/.local/openmpi-4.0.1/bin/mpirun -np 9 -host "gpu10:5,gpu11:4" \
--oversubscribe --prefix /home/esetstore/.local/openmpi-4.0.1 \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include 192.168.0.1/24 \
-x NCCL_DEBUG=INFO  \
-x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
-x NCCL_IB_DISABLE=1 \
/home/zhtang/anaconda3/envs/fedml/bin/python main.py --cf '/home/zhtang/FedML/python/examples/simulation/mpi_torch_fedavg/config/eset.yaml' \
--worker_num 9 --gpu_util_parse 'gpu10:2,1,1,1;gpu11:1,1,1,1' \
--model resnet18_torch  --group_norm_channels 32 \
--federated_optimizer FedAvg --local_cache False  --learning_rate 0.05 --batch_size 20 \
--dataset femnist  --data_cache_dir "/home/zhtang/zhtang_data/datasets" \
--client_num_in_total 3400 --client_num_per_round 1000 --comm_round 100 --epochs 10 \
--frequency_of_the_test 10 \
--aggregate_seq True --hierarchical_agg True \
--runtime_est_mode history \
--tracking_runtime True \
--device_tag eset --tag Ready --exp_name e_fem_c1000_k8_hier --wandb_id e_fem_c1000_k8_hier








