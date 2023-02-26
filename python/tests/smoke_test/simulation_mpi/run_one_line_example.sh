#!/usr/bin/env bash

hostname > mpi_host_file

mpirun --oversubscribe -np 5 \
-hostfile mpi_host_file --oversubscribe \
python main.py --cf fedml_config.yaml