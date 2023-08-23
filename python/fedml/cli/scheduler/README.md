
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
# If your job doesn't contain any source code, it can be empty.
workspace: hello_world

# Running entry commands which will be executed as the job entry point.
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

After the launch CLI is executed, the output is as follows. Here you may open the job url to confirm and actually start the job.
```
Submit your job to the launch platform: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 6.20k/6.20k [00:01<00:00, 3.45kB/s]

Found matched GPU devices for you, which are as follows.
+----------+-------------------+---------+------------+----------------------------+--------+-------+----------+
| Provider |      Instance     | vCPU(s) | Memory(GB) |           GPU(s)           | Region |  Cost | Selected |
+----------+-------------------+---------+------------+----------------------------+--------+-------+----------+
|  FedML   | fedml_a100_node_2 |    4    |     0      | nvidia-fedml_a100_node_2:1 | USA-CA | 60.00 |          |
+----------+-------------------+---------+------------+----------------------------+--------+-------+----------+

Job picasso_dog pre-launch process has started. But the job launch is not started yet.
You may go to this web page with your account to review your job and confirm the launch start.
https://open.fedml.ai/gpu/projects/job/confirmStartJob?projectId=1692900612607447040&projectName=default-project&jobId=1692924448354734080

Or here you can directly confirm to launch your job on the above GPUs.
Are you sure to launch it? [y/N]: y

Currently, your launch result is as follows.
+--------------+---------------------+---------+---------------------+------------+----------+------+
|   Job Name   |        Job ID       |  Status |     Started Time    | Ended Time | Duration | Cost |
+--------------+---------------------+---------+---------------------+------------+----------+------+
| escher_eagle | 1692924497948184576 | RUNNING | 2023-08-19T23:40:27 |    None    |   None   | 0.0  |
+--------------+---------------------+---------+---------------------+------------+----------+------+

You can track your job running details at this URL.
https://open.fedml.ai/gpu/projects/job/jobDetail?projectId=1692900612607447040&jobId=1692924497948184576

For querying the realtime status of your job, please run the following command.
fedml jobs list -id 1692924497948184576

```

## Login as the GPU supplier
If you want to login as the role of GPU supplier and join into the FedML launch payment system. You just need to run the following command.
```
fedml launch login $YourUserId -k $YourApiKey
```

Then you may find your GPU device in the FedML launch platform https://open.fedml.ai/gpu-supplier/gpus/index

And then you may bind your FedML account to your payment account. Once your GPU device is scheduled to run any computing work load, 

you will get some rewards from the GPU consumer with the `fedml launch` CLI.

