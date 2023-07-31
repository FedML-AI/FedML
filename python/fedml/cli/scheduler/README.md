
## Launch jobs with customized commands in the job yaml
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
At first, you need to define your job properties in the job yaml file, e.g. entry file, config file, command arguments, etc.

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

You need to customize the following items (hello_world is an example job).
```
executable_interpreter: python      # shell interpreter for executable_file, e.g. bash, sh, zsh, python, etc.
executable_file_folder: hello_world # directory for executable file
executable_file: job_entry.py     # your main executable file in the executable_file_folder, which can be empty
executable_conf_option: --cf     # your command option for executable_conf_file, which can be empty
executable_conf_file_folder: hello_world/config # directory for config file
executable_conf_file: fedml_config.yaml   # your config file for the main executable program in the executable_conf_file_folder, which can be emptyexecutable_args
```
 
The actual job command will be executed with the following combination.

executable_interpreter executable_file_folder/executable_file executable_conf_option executable_conf_file_folder/executable_conf_file executable_args 

e.g. python hello_world/torch_client.py --cf hello_world/config/fedml_config.yaml --rank 1

e.g. deepspeed <client_entry.py> --deepspeed_config ds_config.json --num_nodes=2 --deepspeed <client args>

e.g. python --version (executable_interpreter=python, executable_args=--version, any else is empty)

e.g. echo "Hello World!" (executable_interpreter=echo, executable_args="Hello World!", any else is empty)

You may use the following example CLI to launch the job at the MLOps platform.
(Replace $YourUserName, $YourUserId, $YourApiKey with your own username, user id and account API key from open.fedml.ai)

Example:
```
fedml launch job call_gpu.yaml -uname $YourUserName -uid $YourUserId -k $YourApiKey
```

After the launch CLI is executed, the output is as follows. Here you may open the job url to confirm and actually start the job.
```
Uploading Package to AWS S3: 100%|██████████| 3.41k/3.41k [00:01<00:00, 2.85kB/s]
Job Cheetah_HelloWorld pre-launch process has started. The job launch is not started yet.
Please go to this web page with your account $YourUserId to review your job and confirm the launch start: {'job_name': None, 'status': None, 'job_url': https://open.fedml.ai/gpu/projects/job/confirmStartJob?projectId=1684824291914420224&jobId=1684833332610863104, 'started_time': 0, 'gpu_matched': None}
For querying the status of the job, please run the command: fedml jobs list -prj Cheetah_HelloWorld -n Cheetah_HelloWorld -u $YourUserId -k $YourApiKey.
```

## Login as the GPU supplier
If you want to login as the role of GPU supplier and join into the FedML Falcon payment system. You just need to run the following command.
```
fedml login $YouUserId -k $YouApiKey -g
```

Then you may find your GPU device in the FedML Falcon platform https://open.fedml.ai/gpu-supplier/gpus/index

After you FedML account bind to your payment account. Once your GPU device is scheduled to run any computing work load, 

you will get some rewards from the GPU consumer with the `fedml launch job` CLI.

