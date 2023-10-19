# Define the package properties
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

# Build the train package with the model and data arguments
fedml train build -sf $SOURCE_FOLDER -ep $ENTRY_FILE -ea "$ENTRY_ARGS" \
  -cf $CONFIG_FOLDER -df $DEST_FOLDER -m $MODEL_NAME -mc $MODEL_CACHE \
  -mi $MODEL_INPUT_DIM -mo $MODEL_OUTPUT_DIM -dn $DATASET_NAME -dt $DATASET_TYPE -dp $DATASET_PATH

# Build the train package without the model and data arguments
# fedml train build -sf $SOURCE_FOLDER -ep $ENTRY_FILE -ea "$ENTRY_ARGS" \
#   -cf $CONFIG_FOLDER -df $DEST_FOLDER