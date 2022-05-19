# FedML MLOps CLI and API Reference

## Overview
```shell
# login to the MLOps Platform
fedml login

# build packages for the MLOps Platform
fedml build

# Logout from the MLOps platform
fedml logout

# Display fedml version
fedml version
```

## 1. Login into the FedML MLOps platform (open.fedml.ai)

```
fedml login userid -v version(release/test)
```

### 1.1. Examples for Logining into the FedML MLOps platform (open.fedml.ai)

```
fedml login 90 
Notes: this will login the production environment for FedML MLOps platform 
```

```
fedml login 90 -v test
Notes: this will login the test environment for FedML MLOps platform 
```

## 2. Build the client and server package in the FedML MLOps platform (open.fedml.ai)

```
fedml build -t client(or server) -sf source_folder -ep entry_point_file -cf config_folder -df destination_package_folder
```

### 2.1. Examples for building the client and server package

```
# build client package
SOURCE=./../cross_silo/client/
ENTRY=torch_client.py
CONFIG=./../cross_silo/config
DEST=./

fedml build -t client \
-sf $SOURCE \
-ep $ENTRY \
-cf $CONFIG \
-df $DEST
```

```
# build server package
SOURCE=./../cross_silo/server/
ENTRY=torch_server.py
CONFIG=./../cross_silo/config
DEST=./

fedml build -t server \
-sf $SOURCE \
-ep $ENTRY \
-cf $CONFIG \
-df $DEST
```

## 3. Logout from the MLOps platform (open.fedml.ai)
```
fedml logout
```

## 4. Display fedml version
```
fedml version
```

You can also refer to a sanity check test example here:
[https://github.com/FedML-AI/FedML/blob/master/test/fedml_user_code/cli/build.sh](https://github.com/FedML-AI/FedML/blob/master/test/fedml_user_code/cli/build.sh)