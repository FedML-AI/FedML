
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
fedml_params:
    fedml_account_id: "111"
    fedml_account_name: "fedml-demo"
    project_name: Cheetah_HelloWorld
    job_name: Cheetah_HelloWorld
    
# Local directory where your source code resides.
work_dir: ~/falcon_examples

# Running entry commands which will be executed as the job entry point.
# Support multiple lines, which can not be empty.
run: | 
    echo "Hello, Here is the Falcon platform."
    echo "Current directory is as follows."
    pwd
    python train.py

# Bootstrap shell commands which will be executed before running entry commands.
# Support multiple lines, which can be empty.
setup: |
  pip install fedml              
  echo "Bootstrap finished."
        
gpu_requirements:
    minimum_num_gpus: 1             # minimum # of GPUs to provision
    maximum_cost_per_hour: $1.75    # max cost per hour for your job per machine
```

You just need to customize the following config items. 

1. `work_dir`, It is the local directory where your source code resides.

2. `run`,  It is the running entry command which will be executed as the job entry point.

3. `setup`, It is the bootstrap shell command which will be executed before running entry commands.

Then you can use the following example CLI to launch the job at the MLOps platform.
(Replace $YourApiKey with your own account API key from open.fedml.ai)

Example:
```
fedml launch call_gpu.yaml -k $YourApiKey
```

After the launch CLI is executed, the output is as follows. Here you may open the job url to confirm and actually start the job.
```
Uploading Package to AWS S3: 100%|██████████| 3.41k/3.41k [00:01<00:00, 2.85kB/s]
Job Cheetah_HelloWorld pre-launch process has started. The job launch is not started yet.
Please go to this web page with your account $YourUserId to review your job and confirm the launch start: {'job_name': None, 'status': None, 'job_url': https://open.fedml.ai/gpu/projects/job/confirmStartJob?projectId=1684824291914420224&jobId=1684833332610863104, 'started_time': 0, 'gpu_matched': None}
For querying the status of the job, please run the command: fedml jobs list -prj Cheetah_HelloWorld -n Cheetah_HelloWorld -u $YourUserId -k $YourApiKey.
```

Notes: 

If your entry program is based on python. We provide logs API to print and upload your printed texts to MLOps.

You may use print or logging.info to print your logs, which will be uploaded to MLOps and can be showed in the logs page.

The example code is as follows.
```
# Init logs before the program starts to log.
mlops.log_print_init()

# Use print or logging.info to print your logs, which will be uploaded to MLOps and can be showed in the logs page.
print("Hello world. Here is the Falcon platform.")
# logging.info("Hello world. Here is the Falcon platform.")

time.sleep(10)

# Cleanup logs when the program will be ended.
mlops.log_print_cleanup()
```

## Login as the GPU supplier
If you want to login as the role of GPU supplier and join into the FedML Falcon payment system. You just need to run the following command.
```
fedml login $YourUserIdOrApiKey -g
```

Then you may find your GPU device in the FedML Falcon platform https://open.fedml.ai/gpu-supplier/gpus/index

And then you may bind your FedML account to your payment account. Once your GPU device is scheduled to run any computing work load, 

you will get some rewards from the GPU consumer with the `fedml launch` CLI.

