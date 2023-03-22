## Run Cross-Machine PyTorch RPC Experiments

Run the following two commands on the caller and callee respectively. Configure the `IFNAME` and `--master_addr` based on your local environment. 

```
GLOO_SOCKET_IFNAME=front0 TP_SOCKET_IFNAME=front0  python multi_machine_launch.py --role=client --comm=ptrpc --master_addr=learnfair100
GLOO_SOCKET_IFNAME=front0 TP_SOCKET_IFNAME=front0  python multi_machine_launch.py --role=server --comm=ptrpc --master_addr=learnfair100

GLOO_SOCKET_IFNAME=eno2 TP_SOCKET_IFNAME=eno2  python multi_machine_launch.py --role=client --comm=ptrpc --master_addr=192.168.1.1
GLOO_SOCKET_IFNAME=eno2 TP_SOCKET_IFNAME=eno2  python multi_machine_launch.py --role=server --comm=ptrpc --master_addr=192.168.1.1
```

## Run Cross-Machine gRPC Experiments

Run the following two commands on the caller and callee respectively. Configure the `--master_addr` based on your local environment. 

```
kill $(ps aux | grep "grpc_client.py" | grep -v grep | awk '{print $2}')
kill $(ps aux | grep "grpc_server.py" | grep -v grep | awk '{print $2}')
python multi_machine_launch.py --role=client --comm=grpc --master_addr=192.168.1.1
python multi_machine_launch.py --role=server --comm=grpc --master_addr=192.168.1.1


python multi_machine_launch.py --role=client --comm=grpc --master_addr=lambda1
python multi_machine_launch.py --role=server --comm=grpc --master_addr=lambda1
```

## Run Single-Machine Experiments

```
python single_machine_launch.py
```
