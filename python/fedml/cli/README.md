
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

## 2. Login into the FedML® Nexus AI Platform (fedml.ai)
login as general computing device with local pip mode:
```
fedml login $YourApiKey
```

login as federated-learning server with local pip mode:
```
fedml login $YourApiKey -s
```

### 2.1. Examples for Logining into the FedML® Nexus AI Platform (fedml.ai)

```
fedml login 113343dad999933 
```

```
fedml login 113343dad999933 -s
```

## 3. Logout from the FedML FedML® Nexus AI Platform (fedml.ai)
logout from computing device with local pip mode:
```
fedml logout 
```

logout from federated-learning server with local pip mode:
```
fedml logout -s
```

## 4. Display fedml version
fedml version


## 5. Display logs
logs from computing device with local pip mode:
```
fedml logs 
```

logs from federated-learning server with local pip mode:
```
fedml logs -s
```

## 6. Diagnosis
Diagnosis for connection to https://open.fedml.ai, AWS S3 and MQTT (mqtt.fedml.ai:1883)
```
fedml diagnosis --open --s3 --mqtt
```

## 7. Jobs
Start a job at FedML® Nexus AI Platform
```
Usage: fedml jobs start [OPTIONS]

Start a job at FedML® Nexus AI Platform

Options:
-pf, --platform TEXT           The platform name at the MLOps platform(options: octopus, parrot, spider, beehive).
-prj, --project_name TEXT      The project name at FedML® Nexus AI Platform
-app, --application_name TEXT  Application name in the My Application list at FedML® Nexus AI Platform
-jn, --job_name TEXT           The job name at FedML® Nexus AI Platform If you don't specify here, the job name from the job yaml file will be used.
-ds, --devices_server TEXT     The server to run the launching job, for the launch platform, we do not need to set this option.
-de, --devices_edges TEXT      The edge devices to run the launching job. Separated with ',', e.g. 705,704. For the launch platform, we do not need to set this option.
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
-k, --api_key TEXT        user api key.
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
    
# Entry commands on multiple OS.
# if running on Windows, the commands in the run_on_windows will be executed.
# if running on posix OS (Mac, Linux, Unix), the commands in the run_on_posix will be executed.
# job:
#   run_on_windows: |
#     echo "I am running on Windows"
#   run_on_posix: |
#     echo "I am running on posix"

# Bootstrap shell commands which will be executed before running entry commands.
# Support multiple lines, which can be empty.
bootstrap: |
  pip install -r requirements.txt
  echo "Bootstrap finished."
  
# Bootstrap shell commands on multiple OS
# if running on Windows, the commands in the run_on_windows will be executed.
# if running on posix OS (Mac, Linux, Unix), the commands in the run_on_posix will be executed.
# bootstrap:
#   run_on_windows: |
#     echo "Bootstrap finished."
#   run_on_posix: |
#     echo "Bootstrap finished."

computing:
  minimum_num_gpus: 1             # minimum # of GPUs to provision

  # max cost per hour of all machines for your job. 
  # E.g., if your job are assigned 2 x A100 nodes (8 GPUs), each GPU cost $1/GPU/Hour, "maximum_cost_per_hour" = 16 * $1 = $16
  maximum_cost_per_hour: $1.75
  
  allow_cross_cloud_resources: false # true, false
  device_type: GPU              # options: GPU, CPU, hybrid
  resource_type: A100-80G       # e.g., A100-80G, please check the resource type list by "fedml show-resource-type" or visiting URL: https://open.fedml.ai/accelerator_resource_type
  
job_type: train              # options: train, deploy, federate
framework_type: fedml        # options: fedml, deepspeed, pytorch, general

# train subtype: general_training, single_machine_training, cluster_distributed_training, cross_cloud_training
# federate subtype: cross_silo, simulation, web, smart_phone
# deploy subtype: none
job_subtype: generate_training

# Running entry commands on the server side which will be executed as the job entry point.
# Support multiple lines, which can not be empty.
server_job: |
    echo "Hello, Here is the server job."
    echo "Current directory is as follows."
    pwd
    
# Entry commands for server jobs on multiple OS.
# if running on Windows, the commands in the run_on_windows will be executed.
# if running on posix OS (Mac, Linux, Unix), the commands in the run_on_posix will be executed.
# server_job:
#   run_on_windows:
#     echo "Hello, Here is the server job on windows."
#   run_on_posix:
#     echo "Hello, Here is the server job on posix."
    
# If you want to use the job created by the MLOps platform,
# just uncomment the following three, then set job_id and config_id to your desired job id and related config.
#job_args:
#  job_id: 2070
#  config_id: 111

# If you want to create the job with specific name, just uncomment the following line and set job_name to your desired job name.
#job_name: cv_job

# If you want to pass your API key to your job for calling FEDML APIs, you may uncomment the following line and set your API key here.
# You may use the environment variable FEDML_RUN_API_KEY to get your API key in your job commands or scripts.
#run_api_key: my_api_key

# If you want to use the model created by the MLOps platform or create your own model card with a specified name,
# just uncomment the following four lines, then set model_name to your desired model name or set your desired endpoint name
#serving_args:
#  model_name: "fedml-launch-sample-model" # Model card from MLOps platform or create your own model card with a specified name
#  model_version: "" # Model version from MLOps platform or set as empty string "" which will use the latest version.
#  endpoint_name: "fedml-launch-endpoint" # Set your end point name which will be deployed, it can be empty string "" which will be auto generated.

# Dataset related arguments
data_args:
  dataset_name: mnist
  dataset_path: ./dataset
  dataset_type: csv
  
# Model related arguments
model_args:
  input_dim: '784'
  model_cache_path: /Users/alexliang/fedml_models
  model_name: lr
  output_dim: '10'
```

You just need to customize the following config items.

1. `workspace`, It is the local directory where your source code resides.

2. `job`,  It is the running entry command which will be executed as the job entry point.

3. `bootstrap`, It is the bootstrap shell command which will be executed before running entry commands.

Then you can use the following example CLI to launch the job at FedML® Nexus AI Platform
(Replace $YourApiKey with your own account API key from open.fedml.ai)

Example:
```
fedml launch hello_job.yaml
```

After the launch CLI is executed, the output is as follows. Here you may open the job url to confirm and actually start the job.
```
Submitting your job to FedML® Nexus AI Platform: 100%|████████████████████████████████████████████████████████████████████████████████████████| 6.07k/6.07k [00:01<00:00, 4.94kB/s]

Searched and matched the following GPU resource for your job:
+-----------+-------------------+---------+------------+-------------------------+---------+-------+----------+
|  Provider |      Instance     | vCPU(s) | Memory(GB) |          GPU(s)         |  Region |  Cost | Selected |
+-----------+-------------------+---------+------------+-------------------------+---------+-------+----------+
| FedML Inc | fedml_a100_node_2 |   256   |   2003.9   | NVIDIA A100-SXM4-80GB:8 | DEFAULT | 40.00 |    √     |
+-----------+-------------------+---------+------------+-------------------------+---------+-------+----------+

You can also view the matched GPU resource with Web UI at: 
https://open.fedml.ai/gpu/projects/job/confirmStartJob?projectId=1692900612607447040&projectName=default-project&jobId=1696947481910317056

Are you sure to launch it? [y/N]: y

Your launch result is as follows:
+------------+---------------------+---------+---------------------+------------------+------+
|  Job Name  |        Job ID       |  Status |       Created       | Spend Time(hour) | Cost |
+------------+---------------------+---------+---------------------+------------------+------+
| munch_clam | 1696947481910317056 | RUNNING | 2023-08-31 02:06:22 |       None       | 0.0  |
+------------+---------------------+---------+---------------------+------------------+------+

You can track your job running details at this URL:
https://open.fedml.ai/gpu/projects/job/jobDetail?projectId=1692900612607447040&jobId=1696947481910317056

For querying the realtime status of your job, please run the following command.
fedml job logs -jid 1696947481910317056
```

### Supported Environment Variables
You may use the following environment variables in your job commands or scripts.
```
$FEDML_CURRENT_RUN_ID, current run id for your job
$FEDML_CURRENT_EDGE_ID, current edge device id for your job
$FEDML_CLIENT_RANK, current device index for your job
$FEDML_CURRENT_VERSION, current fedml config version, options: dev, test or release
$FEDML_RUN_API_KEY, current API key from your job.yaml with the config item run_api_key
```

## 9. Login as the GPU supplier
If you want to login as the role of GPU supplier and join into the FedML launch payment system. You just need to run the following command.
```
fedml login -p $YourApiKey
```

Then you may find your GPU device in the FedML launch platform https://open.fedml.ai/gpu-supplier/gpus/index

And then you may bind your FedML account to your payment account. Once your GPU device is scheduled to run any computing work load,

you will get some rewards from the GPU consumer with the `fedml launch` CLI.


