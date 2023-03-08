#!/usr/bin/env bash

WORKER_NUM=$1
ALG=$2
OPT=$3
PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM


if [ "$ALG" = "gcn" ]; then
    if [ "$OPT" = "fedavg" ]; then
        hostname > mpi_host_file
        mpirun -np $PROCESS_NUM  -hostfile mpi_host_file --oversubscribe \
    python fedml_moleculenet_property_prediction.py --cf config_fedavg/simulation_gcn/fedml_config.yaml
    fi
     if [ "$OPT" = "fedprox" ]; then
     hostname > mpi_host_file
    mpirun -np $PROCESS_NUM -hostfile mpi_host_file --oversubscribe \
python fedml_moleculenet_property_prediction.py --cf config_fedprox/simulation_gcn/fedml_config.yaml
    fi
    if [ "$OPT" = "fedopt" ]; then
    hostname > mpi_host_file
    mpirun -np $PROCESS_NUM -hostfile mpi_host_file --oversubscribe \
    python fedml_moleculenet_property_prediction.py --cf config_fedopt/simulation_gcn/fedml_config.yaml
    fi
fi
if [ "$ALG" = "gat" ]; then
    if [ "$OPT" = "fedavg" ]; then
    hostname > mpi_host_file
    mpirun -np $PROCESS_NUM  -hostfile mpi_host_file --oversubscribe \
python fedml_moleculenet_property_prediction.py --cf config_fedavg/simulation_gat/fedml_config.yaml
    fi
     if [ "$OPT" == "fedprox" ]; then
     hostname > mpi_host_file
    mpirun -np $PROCESS_NUM -hostfile mpi_host_file --oversubscribe \
python fedml_moleculenet_property_prediction.py --cf config_fedprox/simulation_gat/fedml_config.yaml
    fi
    if [ "$OPT" = "fedopt" ]; then
    hostname > mpi_host_file
    mpirun -np $PROCESS_NUM -hostfile mpi_host_file --oversubscribe \
    python fedml_moleculenet_property_prediction.py --cf config_fedopt/simulation_gat/fedml_config.yaml
    fi
fi
if [ "$ALG" = "sage" ]; then
    if [ "$OPT" = "fedavg" ]; then
    hostname > mpi_host_file
    mpirun -np $PROCESS_NUM  -hostfile mpi_host_file --oversubscribe \
python fedml_moleculenet_property_prediction.py --cf config_fedavg/simulation_sage/fedml_config.yaml
    fi
     if [ "$OPT" = "fedprox" ]; then
     hostname > mpi_host_file
    mpirun -np $PROCESS_NUM -hostfile mpi_host_file --oversubscribe \
python fedml_moleculenet_property_prediction.py --cf config_fedprox/simulation_sage/fedml_config.yaml
    fi
    if [ "$OPT" = "fedopt" ]; then
    hostname > mpi_host_file
    mpirun -np $PROCESS_NUM -hostfile mpi_host_file --oversubscribe \
    python fedml_moleculenet_property_prediction.py --cf config_fedopt/simulation_sage/fedml_config.yaml
    fi
fi