
## Build the package for FEDML Federate
```
Usage: fedml federate build [OPTIONS]

  Build federate packages for the FedML® Launch platform (open.fedml.ai).

Options:
  -h, --help                    Show this message and exit.
  -s, --server                  build the server package, default is building
                                client package.
  -sf, --source_folder TEXT     the source code folder path
  -ep, --entry_point TEXT       the entry point of the source code
  -ea, --entry_args TEXT        entry arguments of the entry point program
  -cf, --config_folder TEXT     the config folder path
  -df, --dest_folder TEXT       the destination package folder path
  -ig, --ignore TEXT            the ignore list for copying files, the format
                                is as follows: *.model,__pycache__,*.data*,
  -m, --model_name TEXT         model name for training.
  -mc, --model_cache_path TEXT  model cache path for training.
  -mi, --input_dim TEXT         input dimensions for training.
  -mo, --output_dim TEXT        output dimensions for training.
  -dn, --dataset_name TEXT      dataset name for training.
  -dt, --dataset_type TEXT      dataset type for training.
  -dp, --dataset_path TEXT      dataset path for training.
```

At first, you need to define your package properties as follows.
If you want to ignore some folders or files, you may specify the ignore argument 
or add them to the .gitignore file in the source code folder.   

### Required arguments:
source code folder, entry file, entry arguments,
config folder, built destination folder

### Optional arguments:
You may define the model and data arguments using the command arguments as follows.
```
model name, model cache path, model input dimension, model output dimension,
dataset name, dataset type, dataset path.
```

Also, you may define the model and data arguments using the file named fedml_config.yaml as follows.
```
fedml_data_args:
    dataset_name: mnist
    dataset_path: ./dataset
    dataset_type: csv
    
fedml_model_args:
    input_dim: '784'
    model_cache_path: /Users/alexliang/fedml_models
    model_name: lr
    output_dim: '10'
```

The above model and data arguments will be mapped to the equivalent environment variables as follows.
```
dataset_name = $FEDML_DATASET_NAME
dataset_path = $FEDML_DATASET_PATH
dataset_type = $FEDML_DATASET_TYPE
model_name = $FEDML_MODEL_NAME
model_cache_path = $FEDML_MODEL_CACHE_PATH
input_dim = $FEDML_MODEL_INPUT_DIM
output_dim = $FEDML_MODEL_OUTPUT_DIM
```

Your may pass these environment variables as your entry arguments. e.g.,
```
ENTRY_ARGS_MODEL_DATA='-m $FEDML_MODEL_NAME -mc $FEDML_MODEL_CACHE_PATH -mi $FEDML_MODEL_INPUT_DIM -mo $FEDML_MODEL_OUTPUT_DIM -dn $FEDML_DATASET_NAME -dt $FEDML_DATASET_TYPE -dp $FEDML_DATASET_PATH'
```

### Examples
```
# Define the federated package properties
SOURCE_FOLDER=.
ENTRY_FILE=train.py
ENTRY_ARGS='--epochs 1'
ENTRY_ARGS_MODEL_DATA='-m $FEDML_MODEL_NAME -mc $FEDML_MODEL_CACHE_PATH -mi $FEDML_MODEL_INPUT_DIM -mo $FEDML_MODEL_OUTPUT_DIM -dn $FEDML_DATASET_NAME -dt $FEDML_DATASET_TYPE -dp $FEDML_DATASET_PATH'
CONFIG_FOLDER=config
DEST_FOLDER=./mlops
MODEL_NAME=lr
MODEL_CACHE=~/fedml_models
MODEL_INPUT_DIM=784
MODEL_OUTPUT_DIM=10
DATASET_NAME=mnist
DATASET_TYPE=csv
DATASET_PATH=./dataset

# Build the federated client package with the model and data arguments
fedml federate build -sf $SOURCE_FOLDER -ep $ENTRY_FILE -ea "$ENTRY_ARGS" \
  -cf $CONFIG_FOLDER -df $DEST_FOLDER \
  -m $MODEL_NAME -mc $MODEL_CACHE -mi $MODEL_INPUT_DIM -mo $MODEL_OUTPUT_DIM \
  -dn $DATASET_NAME -dt $DATASET_TYPE -dp $DATASET_PATH

# Build the federated client package without the model and data arguments
# fedml federate build -sf $SOURCE_FOLDER -ep $ENTRY_FILE -ea "$ENTRY_ARGS" \
#  -cf $CONFIG_FOLDER -df $DEST_FOLDER 
 
# Define the federated server package properties
ENTRY_FILE=torch_server.py

# Build the federated server package with the model and data arguments
fedml federate build -s -sf $SOURCE_FOLDER -ep $ENTRY_FILE -ea "$ENTRY_ARGS" \
  -cf $CONFIG_FOLDER -df $DEST_FOLDER \
  -m $MODEL_NAME -mc $MODEL_CACHE -mi $MODEL_INPUT_DIM -mo $MODEL_OUTPUT_DIM \
  -dn $DATASET_NAME -dt $DATASET_TYPE -dp $DATASET_PATH
  
# Build the federated server package without the model and data arguments
# fedml federate build -s -sf $SOURCE_FOLDER -ep $ENTRY_FILE -ea "$ENTRY_ARGS" \
#  -cf $CONFIG_FOLDER -df $DEST_FOLDER
```
