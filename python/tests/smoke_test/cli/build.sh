# https://doc.fedml.ai/mlops/api.html

#fedml build -t client -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST
#
#Usage: fedml build [OPTIONS]
#
#  Commands for open.fedml.ai MLOps platform
#
#Options:
#  -t, --type TEXT            client or server? (value: client; server)
#  -sf, --source_folder TEXT  the source code folder path
#  -ep, --entry_point TEXT    the entry point of the source code
#  -cf, --config_folder TEXT  the config folder path
#  -df, --dest_folder TEXT    the destination package folder path
#  --help                     Show this message and exit.

# build client package
cd ../../../../examples/cross_silo/mqtt_s3_fedavg_mnist_lr_example/one_line

SOURCE=client
ENTRY=torch_client.py
CONFIG=config
DEST=./mlops
fedml build -t client -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST

# build server package
SOURCE=server
ENTRY=torch_server.py
CONFIG=config
DEST=./mlops
fedml build -t server -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST