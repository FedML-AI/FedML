# FedML MLOps CLI and API Reference

## Overview
```shell
# login to the MLOps Platform
fedml login

# build packages for the MLOps Platform
fedml build

```


## 1. Login into the FedML MLOps platform (open.fedml.ai)

```
fedml login userid -v version(release/test)
```

### 1.1. Examples for Logining into the FedML MLOps platform (open.fedml.ai)

```
fedml login 90
```

```
fedml login 90 -v test
```

## 2. Build the client and server package in the FedML MLOps platform (open.fedml.ai)

```
fedml build -t client(or server) -sf source_folder -ep entry_point_file -cf config_folder -df destination_package_folder
```

### 2.1. Examples for building the client and server package

```
fedml build \
    -t client \
    -sf /Users/alexliang/Work/my-docs/inc-project/FM-Proj/Src/FedML-refactor/python/examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example/one_line/client \
    -ep torch_client.py \
    -cf /Users/alexliang/Work/my-docs/inc-project/FM-Proj/Src/FedML-refactor/python/examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example/one_line/config \
    -df /Users/alexliang/fedml-test
```

```
fedml build \
    -t server \
    -sf /Users/alexliang/Work/my-docs/inc-project/FM-Proj/Src/FedML-refactor/python/examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example/one_line/server \
    -ep torch_server.py \
    -cf /Users/alexliang/Work/my-docs/inc-project/FM-Proj/Src/FedML-refactor/python/examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example/one_line/config \
    -df /Users/alexliang/fedml-test
```
