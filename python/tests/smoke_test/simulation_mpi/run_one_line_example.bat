
hostname > mpi_host_file

mpirun --oversubscribe -np 5 -hostfile mpi_host_file python main.py --cf fedml_config.yaml