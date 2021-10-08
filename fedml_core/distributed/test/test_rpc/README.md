

### Scripts

#### gRPC
```
# server
sh run_rpc.sh GRPC 0


# client
sh run_rpc.sh GRPC 1
```


#### Torch RPC
```
# server
sh run_rpc.sh TRPC 0


# client
sh run_rpc.sh TRPC 1
```

#### GRPC configuration
```USC lambda 1&4
receiver_id,ip
0,68.181.2.242
1,10.136.200.90
```

```AWS cross-acount machines
receiver_id,ip
0,10.1.4.147
1,10.1.4.186
2,10.2.2.158
```

### TRPC configuration
``` USC lambda 1&4
master_ip,master_port
68.181.2.242,9999
```

```AWS cross-acount machines
master_ip,master_port
10.1.4.147,9999
```

