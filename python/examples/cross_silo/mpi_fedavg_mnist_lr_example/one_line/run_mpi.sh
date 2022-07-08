# ~/.conda/envs/rpc-benchmark/bin/mpirun --mca btl_tcp_if_include 10.0.0.0/8 -np 4 -hostfile ./mpi_host_file python3 ./torch_mpi.py --cf ./config/fedml_config.yaml --rank 0
WORKER_NUM=$1

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

$(which mpirun) -np $PROCESS_NUM \
    -hostfile ./config/mpi_host_file \
    $(which python3) ./torch_mpi.py --cf ./config/fedml_config.yaml
