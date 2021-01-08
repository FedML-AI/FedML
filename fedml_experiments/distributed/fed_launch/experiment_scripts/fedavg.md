# FedAvg Example 


```
# Remember to change the python path, hostfile, gpu_util_file, gpu_util_key, and data_dir for your configuration.
# If gpu_util_file is None, you will use CPU.

# This script will run fedavg with 11 process, in which, 1 server process and 10 client processes.
# The GPU mapping is defined in `gpuutils/local_gpu_util.yaml -- config4_11`

sh run_fedavg.sh 10 hostfiles/gpuhome_mpi_host_file ~/py36/bin/python " --entity automl --project fedml --algorithm FedAvg --gpu_server_num 1 --gpu_num_per_server 1 --model mobilenet --dataset cifar10 --data_dir /home/user/data/cifar10  --partition_method hetero --client_num_in_total 10 --comm_round 90  --epochs 1 --client_optimizer sgd --batch_size 100 --lr 0.001 --wd 0.0001 --momentum 0.9 --ci 0  --gpu_util_file gpuutils/local_gpu_util.yaml --gpu_util_key config4_11"
```
