# Define the federated package properties
SOURCE_FOLDER=.
ENTRY_FILE=torch_client.py
ENTRY_ARGS='-m $FEDML_MODEL_NAME -mc $FEDML_MODEL_CACHE_PATH -mi $FEDML_MODEL_INPUT_DIM -mo $FEDML_MODEL_OUTPUT_DIM -dn $FEDML_DATASET_NAME -dt $FEDML_DATASET_TYPE -dp $FEDML_DATASET_PATH'
CONFIG_FOLDER=config
DEST_FOLDER=./mlops
MODEL_NAME=lr
MODEL_CACHE=~/fedml_models
MODEL_INPUT_DIM=784
MODEL_OUTPUT_DIM=10
DATASET_NAME=mnist
DATASET_TYPE=csv
DATASET_PATH=~/fedml_data

# Build the federated client package with the model and data arguments
fedml federate build -sf $SOURCE_FOLDER -ep $ENTRY_FILE -ea "$ENTRY_ARGS" \
  -cf $CONFIG_FOLDER -df $DEST_FOLDER \
  -m $MODEL_NAME -mc $MODEL_CACHE -mi $MODEL_INPUT_DIM -mo $MODEL_OUTPUT_DIM \
  -dn $DATASET_NAME -dt $DATASET_TYPE -dp $DATASET_PATH

# Build the federated client package without the model and data arguments
#fedml federate build -sf $SOURCE_FOLDER -ep $ENTRY_FILE -ea "$ENTRY_ARGS" \
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