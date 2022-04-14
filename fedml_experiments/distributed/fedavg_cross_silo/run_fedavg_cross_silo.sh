#!/usr/bin/env bash

arguments=""


NETWORK_INTERFACE=lo

while [ $# -gt 0 ]; do
  case "$1" in
    --network_interface)
      NETWORK_INTERFACE="${2}"
      ;;
    --nproc_per_node)
      NPROC_PER_NODE="${2}"
      arguments="$arguments $1 $2"
      ;;
    --enable_cuda_rpc)
      arguments="$arguments $1"
      ;;
    --*)
      arguments="$arguments $1 $2"
  esac
  shift
done


export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
export GLOO_SOCKET_IFNAME=$NETWORK_INTERFACE
export TP_SOCKET_IFNAME=$NETWORK_INTERFACE

export NCCL_DEBUG=INFO
export NCCL_MIN_NRINGS=1
export NCCL_TREE_THRESHOLD=4294967296
export OMP_NUM_THREADS=8
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_SOCKET_NTHREADS=8
export NCCL_BUFFSIZE=1048576


(mpirun -np $NPROC_PER_NODE ./main_fedavg_cross_silo.py \
  $arguments
) 



