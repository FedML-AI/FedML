
## Launch jobs with customized commands in the job yaml
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

# Bootstrap shell commands which will be executed before running entry commands.
# Support multiple lines, which can be empty.
bootstrap: |
  pip install -r requirements.txt
  echo "Bootstrap finished."

computing:
  minimum_num_gpus: 1             # minimum # of GPUs to provision

  # max cost per hour of all machines for your job. 
  # E.g., if your job are assigned 2 x A100 nodes (8 GPUs), each GPU cost $1/GPU/Hour, "maximum_cost_per_hour" = 16 * $1 = $16
  maximum_cost_per_hour: $1.75
  
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
fedml launch hello_job.yaml
```

After the launch CLI is executed, the output is as follows. Here you may open the job url to confirm and actually start the job.
```
Submitting your job to FedML® Launch platform: 100%|████████████████████████████████████████████████████████████████████████████████████████| 6.07k/6.07k [00:01<00:00, 4.94kB/s]

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
fedml launch log 1696947481910317056
```

## Login as the GPU supplier
If you want to login as the role of GPU supplier and join into the FedML launch payment system. You just need to run the following command.
```
fedml login $YourUserId -k $YourApiKey -r gpu_supplier
```

Then you may find your GPU device in the FedML launch platform https://open.fedml.ai/gpu-supplier/gpus/index

And then you may bind your FedML account to your payment account. Once your GPU device is scheduled to run any computing work load, 

you will get some rewards from the GPU consumer with the `fedml launch` CLI.

