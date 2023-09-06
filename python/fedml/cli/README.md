
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
-jn, --job_name TEXT           The job name at the MLOps platform. If you don't specify here, the job name from the job yaml file will be used.
-ds, --devices_server TEXT     The server to run the launching job, for the launch platform, we do not need to set this option.
-de, --devices_edges TEXT      The edge devices to run the launching job. Seperated with ',', e.g. 705,704. For the launch platform, we do not need to set this option.
-u, --user TEXT                user id or api key.
-k, --api_key TEXT             user api key.
-v, --version TEXT             start job at which version of MLOps platform. It should be dev, test or release
--help                         Show this message and exit.
```

Example: 
```
fedml jobs start -pf octopus -prj test-fedml -app test-alex-app -ds 706 -de 705,704 -u 214 -k c9356b9c4ce44363bb66366d210301
```

## 8. Launch jobs with customized commands in the job yaml
```
Usage: fedml launch [OPTIONS] [YAML_FILE]...

launch job at the MLOps platform

Options:
-uname, --user_name TEXT  user name. If you do not specify this option, the fedml_account_name field from YAML_FILE will be used.
-uid, --user_id TEXT      user id. If you do not specify this option, the fedml_account_id field from YAML_FILE will be used.
-k, --api_key TEXT        user api key.
-pf, --platform TEXT      The platform name at the MLOps platform (options:octopus, parrot, spider, beehive, launch, default is launch).
-jn, --job_name TEXT      The job name at the MLOps platform. If you don't specify here, the job name from the job yaml file will be used.
-ds, --devices_server TEXT  The server to run the launching job, for the launch platform, we do not need to set this option.
-de, --devices_edges TEXT   The edge devices to run the launching job. Seperated with ',', e.g. 705,704. For the launch platform, we do not need to set this option.
-nc, --no_confirmation    no confirmation after initiating launching request.
-v, --version TEXT        launch job to which version of MLOps platform. It should be dev, test or release
--help                    Show this message and exit.
```
At first, you need to define your job properties in the job yaml file, e.g. entry file, config file, command arguments, etc.

The job yaml file is as follows:
```
fedml_env:
  project_name: 

# Local directory where your source code resides.
# It should be the relative path to this job yaml file or the absolute path.
# If your job doesn't contain any source code, it can be empty.
workspace: hello_world

# Running entry commands which will be executed as the job entry point.
# If an error occurs, you should exit with a non-zero code, e.g. exit 1.
# Otherwise, you should exit with a zero code, e.g. exit 0.
# Support multiple lines, which can not be empty.
job: | 
    echo "Hello, Here is the launch platform."
    echo "Current directory is as follows."
    pwd
    python hello_world.py

# Bootstrap shell commands which will be executed before running entry commands.
# Support multiple lines, which can be empty.
bootstrap: |
  pip install -r requirements.txt
  echo "Bootstrap finished."

computing:
  minimum_num_gpus: 1             # minimum # of GPUs to provision
  maximum_cost_per_hour: $1.75    # max cost per hour for your job per machine
  allow_cross_cloud_resources: false # true, false
  device_type: GPU              # options: GPU, CPU, hybrid
  resource_type: A100-80G       # e.g., A100-80G, please check the resource type list by "fedml show-resource-type" or visiting URL: https://open.fedml.ai/accelerator_resource_type
  
framework_type: fedml         # options: fedml, deepspeed, pytorch, general
task_type: train              # options: serve, train, dev-environment

# Running entry commands on the server side which will be executed as the job entry point.
# Support multiple lines, which can not be empty.
server_job: |
    echo "Hello, Here is the server job."
    echo "Current directory is as follows."
    pwd
```

You just need to customize the following config items.

1. `workspace`, It is the local directory where your source code resides.

2. `job`,  It is the running entry command which will be executed as the job entry point.

3. `bootstrap`, It is the bootstrap shell command which will be executed before running entry commands.

Then you can use the following example CLI to launch the job at the MLOps platform.
(Replace $YourApiKey with your own account API key from open.fedml.ai)

Example:
```
fedml launch call_gpu.yaml
```
