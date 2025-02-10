
## Build the package for FEDML Train
```
Usage: fedml train build [OPTIONS] [YAML_FILE]

  Build training packages for the TensorOpera® Nexus AI Platform.

Options:
  -h, --help              Show this message and exit.
  -d, --dest_folder TEXT  The destination package folder path. If this option
                          is not specified, the built packages will be located
                          in a subdirectory named fedml-train-packages in the
                          directory of YAML_FILE
```

At first, you need to define your job properties in the job yaml file, e.g., workspace, job entry commands.

This job yaml file can be from the launch job yaml file, which is as follows:

```
# Local directory where your source code resides.
# It should be the relative path to this job yaml file or the absolute path.
# If your job doesn't contain any source code, it can be empty.
workspace: .

# Running entry commands which will be executed as the job entry point.
# If an error occurs, you should exit with a non-zero code, e.g. exit 1.
# Otherwise, you should exit with a zero code, e.g. exit 0.
# Support multiple lines, which can not be empty.
job: |
    python3 train.py --epochs 1

job_type: train              # options: train, deploy, federate

# Bootstrap shell commands which will be executed before running entry commands.
# Support multiple lines, which can be empty.
bootstrap: |
  echo "Bootstrap finished."

computing:
  minimum_num_gpus: 1           # minimum # of GPUs to provision
  maximum_cost_per_hour: $3000   # max cost per hour for your job per gpu card
  #allow_cross_cloud_resources: true # true, false
  #device_type: CPU              # options: GPU, CPU, hybrid
  resource_type: A100-80G       # e.g., A100-80G, please check the resource type list by "fedml show-resource-type" or visiting URL: https://open.fedml.ai/accelerator_resource_type

data_args:
  dataset_name: mnist
  dataset_path: ./dataset
  dataset_type: csv

model_args:
  input_dim: '784'
  model_cache_path: /Users/alexliang/fedml_models
  model_name: lr
  output_dim: '10'

training_params:
  learning_rate: 0.004
```

The config items will be mapped to the equivalent environment variables with the following rules.

if the config path for config_sub_key is as follows.
```
config_parent_key:
    config_sub_key: config_sub_key_value
```

Then the equivalent environment variable will be as follows.

```
FEDML_ENV_uppercase($config_parent_key)_uppercase($config_sub_key)
```

e.g., the equivalent environment variables of above example config items will be as follows. 

```
dataset_name = $FEDML_ENV_DATA_ARGS_DATASET_NAME
dataset_path = $FFEDML_ENV_DATA_ARGS_DATASET_PATH
dataset_type = $FEDML_ENV_DATA_ARGS_DATASET_TYPE
model_name = $FEDML_ENV_MODEL_ARGS_MODEL_NAME
model_cache_path = $FEDML_ENV_MODEL_ARGS_MODEL_CACHE_PATH
input_dim = $FEDML_ENV_MODEL_ARGS_MODEL_INPUT_DIM
output_dim = $FEDML_ENV_MODEL_ARGS_MODEL_OUTPUT_DIM
```

Your may use these environment variables in your job commands. e.g.,
```
job: |
    python3 train.py --epochs 1 -m $FEDML_ENV_MODEL_ARGS_MODEL_NAME -mc $FEDML_ENV_MODEL_ARGS_MODEL_CACHE_PATH -mi $FEDML_ENV_MODEL_ARGS_MODEL_INPUT_DIM -mo $FEDML_ENV_MODEL_ARGS_MODEL_OUTPUT_DIM -dn $FEDML_ENV_DATA_ARGS_DATASET_NAME -dt $FEDML_ENV_DATA_ARGS_DATASET_TYPE -dp $FEDML_ENV_DATA_ARGS_DATASET_PATH
```

### Examples
```
fedml train build train_job.yaml

Your train package file is located at: /home/fedml/launch/fedml-train-packages/client-package.zip
```
