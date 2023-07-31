
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
-pf, --platform TEXT           The platform name at the MLOps platform(options: octopus, parrot, spider, beehive).
-prj, --project_name TEXT      The project name at the MLOps platform.
-app, --application_name TEXT  Application name in the My Application list at the MLOps platform.
-d, --devices TEXT             The devices with the format: [{"serverId":727, "edgeIds": ["693"], "account": 105}]
-u, --user TEXT                user id or api key.
-k, --api_key TEXT             user api key.
-v, --version TEXT             start job at which version of MLOps platform. It should be dev, test or release
--help                         Show this message and exit.
```

Example: 
```
fedml jobs start -pf octopus -prj test-fedml -app test-alex-app -d '[{"serverId":706,"edgeIds":["705"],"account":214}]' -u 214 -k c9356b9c4ce44363bb66366d210301
```

## 8. Launch jobs with customized commands in the job yaml
```
Usage: fedml launch job [OPTIONS] [YAML_FILE]...

launch job at the MLOps platform

Options:
-uname, --user_name TEXT  user name.
-uid, --user_id TEXT      user id.
-k, --api_key TEXT        user api key.
-pf, --platform TEXT      The platform name at the MLOps platform (options:octopus, parrot, spider, beehive, falcon).
-d, --devices TEXT        The devices with the format: [{"serverId": 727,"edgeIds": ["693"], "account": 105}]
-nc, --no_confirmation    no confirmation after initiating launching request.
-v, --version TEXT        launch job to which version of MLOps platform. It should be dev, test or release
--help                    Show this message and exit.
```
After you define your job properties in the job yaml file, e.g. entry file, config file, command arguments, etc.

The job yaml file is as follows:
```
fedml_params:
    fedml_account_id: 1111
    project_name: Cheetah_HelloWorld
    job_name: Cheetah_HelloWorld13

development_resources:
    dev_env: "https://open.fedml.ai"  # development resources bundle to load on each machine
    network: mqtt_s3    # network protocol for communication between machines

executable_code_and_data:
    # The entire command will be executed as follows:
    # executable_interpreter executable_file_folder/executable_file executable_conf_option executable_conf_file_folder/executable_conf_file executable_args
    # e.g. python hello_world/torch_client.py --cf hello_world/config/fedml_config.yaml --rank 1
    # e.g. deepspeed <client_entry.py> --deepspeed_config ds_config.json --num_nodes=2 --deepspeed <client args>
    # e.g. python --version (executable_interpreter=python, executable_args=--version, any else is empty)
    # e.g. echo "Hello World!" (executable_interpreter=echo, executable_args="Hello World!", any else is empty)
    executable_interpreter: python                   # shell interpreter for executable_file, e.g. bash, sh, zsh, python, etc.
    executable_file_folder: hello_world # directory for executable file
    executable_file: job_entry.py     # your main executable file in the executable_file_folder, which can be empty
    executable_conf_option: --cf     # your command option for executable_conf_file, which can be empty
    executable_conf_file_folder: hello_world/config # directory for config file
    executable_conf_file: fedml_config.yaml   # your config file for the main executable program in the executable_conf_file_folder, which can be empty
    executable_args: --rank 1            # command arguments for the executable_interpreter and executable_file
    data_location: ~/fedml_data          # path to your data
    # bootstrap shell commands which will be executed before running executable_file. support multiple lines, which can be empty
    bootstrap: |
    ls -la ~               
    echo "Bootstrap..."
gpu_requirements:
    minimum_num_gpus: 1             # minimum # of GPUs to provision
    maximum_cost_per_hour: $1.75    # max cost per hour for your job per machine
```

You may use the above CLI to launch the job at the MLOps platform. 

Example:
```
fedml launch job tin_pond.yaml -uname $YourUserName -uid $YourUserId -k $YourApiKey
```
