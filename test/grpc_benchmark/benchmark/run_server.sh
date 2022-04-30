python multi_machine_launch.py --role=server --comm=ptrpc
python multi_machine_launch.py --role=server --comm=grpc

# GLOO_SOCKET_IFNAME=enp1s0f0 python multi_machine_launch.py --role=client --comm=ptrpc --master_addr=learnfair088
# $(ip r | grep default | awk '{print $5}')

# GLOO_SOCKET_IFNAME=front0 TP_SOCKET_IFNAME=front0  python multi_machine_launch.py --role=client --comm=ptrpc --master_addr=learnfair100
# GLOO_SOCKET_IFNAME=front0 TP_SOCKET_IFNAME=front0  python multi_machine_launch.py --role=server --comm=ptrpc --master_addr=learnfair100

# python multi_machine_launch.py --role=client --comm=grpc --master_addr=100.97.17.9
# python multi_machine_launch.py --role=server --comm=grpc --master_addr=100.97.17.9
