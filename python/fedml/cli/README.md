
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

login as edge server with local pip mode:
```
fedml login userid(or API Key) -s
```

login as edge server with docker mode:
```
fedml login userid(or API Key) -s --docker --docker-rank 1
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

logout from edge server with local pip mode:
```
fedml logout -s
```

logout from edge server with docker mode:
```
fedml logout -s --docker --docker-rank 1
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

logs from edge server with local pip mode:
```
fedml logs -s
```

logs from edge server with docker mode:
```
fedml logs --docker --docker-rank 1
```

## 6. Diagnosis
Diagnosis for connection to https://open.fedml.ai, AWS S3 and MQTT (mqtt.fedml.ai:1883)
```
fedml diagnosis --open --s3 --mqtt
```

## 7. Jobs
Start a job at the MLOps platform.
```
Usage: fedml jobs start [OPTIONS]

Start a job at the MLOps platform.

Options:
-pf, --platform TEXT           The platform name at the MLOps platform
(options: octopus, parrot, spider, beehive).
-prj, --project_name TEXT      The project name at the MLOps platform.
-app, --application_name TEXT  Application name in the My Application list
at the MLOps platform.
-d, --devices TEXT             The devices with the format: [{"serverId":
727, "edgeIds": ["693"], "account": 105}]
-u, --user TEXT                user id or api key.
-k, --api_key TEXT             user api key.
-v, --version TEXT             start job at which version of MLOps platform.
It should be dev, test or release
--help                         Show this message and exit.
```

Example: 
```
fedml jobs start -pf octopus -prj test-fedml -app test-alex-app -d '[{"serverId":706,"edgeIds":["705"],"account":214}]' -u 214 -k c9356b9c4ce44363bb66366d210301
```