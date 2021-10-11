# Benchmarking


## 1. Run Test Scripts

### gRPC

configure `./grpc/mpi_host_file`, `./grpc/gpu_mapping.yaml` and `./grpc/grpc_ipconfig.csv`.

Run gRPC tests.
```
cd grpc && run_grpc.sh
```

Logs are stored in `./grpc/logs`. Each log's filename contains execution timestamp.

### tRPC

configure `./trpc/mpi_host_file`, `./trpc/gpu_mapping.yaml` and `./trpc/trpc_master_config.csv`.

Run tRPC tests with cuda rpc enabled
```
cd trpc && sh run_trpc_cuda_rpc_enabled.sh
```

Run tRPC tests with cuda rpc disabled
```
cd trpc && sh run_trpc_cuda_rpc_disabled.sh
```

Logs are stored in `./trpc/logs`. Each log's filename contains execution timestamp.


## 2. Parse Logs
Parse logs
```
python log-parser.py ./path/to/log
```

