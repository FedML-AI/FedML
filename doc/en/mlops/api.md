# FedML MLOps CLI and API Reference

## Overview
```shell
# log in to the MLOps Platform
fedml login

# build packages for the MLOps Platform
fedml build

# Logout from the MLOps platform
fedml logout

# Display FedML environment
fedml env

# Display FedML version
fedml version
```

## 1. Log in to the FedML MLOps platform (open.fedml.ai)
login as client with local pip mode:
```
fedml login userid(or API Key)
```

login as client with docker mode:
```
fedml login userid(or API Key) --docker --docker-rank rank_index
```

login as edge server with local pip mode:
```
fedml login userid(or API Key) -s
```

login as edge server with docker mode:
```
fedml login userid(or API Key) -s --docker --docker-rank rank_index
```

### 1.1. Examples for Logging in to the FedML MLOps platform (open.fedml.ai)

```
fedml login 90 
Notes: this will login the production environment for FedML MLOps platform 
```

```
fedml login 90 --docker --docker-rank 1
Notes: this will login the production environment with docker mode for FedML MLOps platform
```

## 2. Build the client and server package in the FedML MLOps platform (open.fedml.ai)

```
fedml build -t client(or server) -sf source_folder -ep entry_point_file -cf config_folder -df destination_package_folder --ignore ignore_file_and_directory(concat with ,)
```

### 2.1. Examples for building the client and server package

```
# build client package
SOURCE=./../cross_silo/client/
ENTRY=torch_client.py
CONFIG=./../cross_silo/config
DEST=./
IGNORE=__pycache__,*.git

fedml build -t client \
-sf $SOURCE \
-ep $ENTRY \
-cf $CONFIG \
-df $DEST
--ignore $IGNORE
```

```
# build server package
SOURCE=./../cross_silo/server/
ENTRY=torch_server.py
CONFIG=./../cross_silo/config
DEST=./
IGNORE=__pycache__,*.git

fedml build -t server \
-sf $SOURCE \
-ep $ENTRY \
-cf $CONFIG \
-df $DEST
--ignore $IGNORE
```

## 3. Log out the MLOps platform (open.fedml.ai)
logout from client with local pip mode:
```
fedml logout 
```

logout from client with docker mode:
```
fedml logout --docker --docker-rank 1
```

logout from edge server with local pip mode:
```
fedml logout -s
```

logout from edge server with docker mode:
```
fedml logout -s --docker --docker-rank 1
```

## 4. Display FedML Environment and Version
```
fedml env
fedml version
```

## 5. Display logs
logs from client with local pip mode:
```
fedml logs 
```

logs from client with docker mode:
```
fedml logs --docker --docker-rank 1
```

logs from edge server with local pip mode:
```
fedml logs -s
```

logs from edge server with docker mode:
```
fedml logs --docker --docker-rank 1
```

You can also refer to a sanity check test example here:
[https://github.com/FedML-AI/FedML/blob/master/test/fedml_user_code/cli/build.sh](https://github.com/FedML-AI/FedML/blob/master/test/fedml_user_code/cli/build.sh)