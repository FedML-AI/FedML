
## 1. Build the client and server package in the FedML MLOps platform (open.fedml.ai)

```
fedml build -t client(or server) -sf source_folder -ep entry_point_file -cf config_folder -df destination_package_folder --ignore __pycache__,*.git
```

### 1.1. Examples for building the client and server package

```
fedml build \
    -t client \
    -sf /Users/alexliang/Work/my-docs/inc-project/FM-Proj/Src/FedML-refactor/python/examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example/one_line/client \
    -ep torch_client.py \
    -cf /Users/alexliang/Work/my-docs/inc-project/FM-Proj/Src/FedML-refactor/python/examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example/one_line/config \
    -df /Users/alexliang/fedml-test
    --ignore __pycache__,*.git
```

```
fedml build \
    -t server \
    -sf /Users/alexliang/Work/my-docs/inc-project/FM-Proj/Src/FedML-refactor/python/examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example/one_line/server \
    -ep torch_server.py \
    -cf /Users/alexliang/Work/my-docs/inc-project/FM-Proj/Src/FedML-refactor/python/examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example/one_line/config \
    -df /Users/alexliang/fedml-test
    --ignore __pycache__,*.git
```

## 2. Login into the FedML MLOps platform (open.fedml.ai)
login as client with local pip mode:
```
fedml login userid(or API Key)
```

login as client with docker mode:
```
fedml login userid(or API Key) --docker --docker-rank 1
```

login as edge server
```
fedml login userid(or API Key) -s
```

### 2.1. Examples for Logining into the FedML MLOps platform (open.fedml.ai)

```
fedml login 90 
```

```
fedml login 90 -s
```

## 3. Logout from the FedML MLOps platform (open.fedml.ai)
logout from client with local pip mode:
```
fedml logout 
```

logout from client with docker mode:
```
fedml logout --docker --docker-rank 1
```

logout from edge server:
```
fedml logout -s
```

## 4. Display fedml version
fedml version


## 5. Display logs
logs from client with local pip mode:
```
fedml logs 
```

logs from client with docker mode:
```
fedml logs --docker --docker-rank 1
```

logs from edge server:
```
fedml logs -s
```