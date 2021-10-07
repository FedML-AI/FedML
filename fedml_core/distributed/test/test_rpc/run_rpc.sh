# enable InfiniBand
#export NCCL_SOCKET_IFNAME=ib0
#export GLOO_SOCKET_IFNAME=ib0
#export TP_SOCKET_IFNAME=ib0
#export NCCL_IB_HCA=ib0

# disable InfiniBand
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eno2
export GLOO_SOCKET_IFNAME=eno2
export TP_SOCKET_IFNAME=eno2

export NCCL_DEBUG=INFO
export NCCL_MIN_NRINGS=1
export NCCL_TREE_THRESHOLD=4294967296
export OMP_NUM_THREADS=8
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_SOCKET_NTHREADS=8
export NCCL_BUFFSIZE=1048576



#kill $(ps aux | grep "main.py" | grep -v grep | awk '{print $2}')
python3 ./main.py \
--backend $1 \
--rank $2