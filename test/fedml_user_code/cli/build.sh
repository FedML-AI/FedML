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
SOURCE=./../cross_silo/client/
ENTRY=torch_client.py
CONFIG=./../cross_silo/config
DEST=./

fedml build -t client \
-sf $SOURCE \
-ep $ENTRY \
-cf $CONFIG \
-df $DEST

# build server package
SOURCE=./../cross_silo/server/
ENTRY=torch_server.py
CONFIG=./../cross_silo/config
DEST=./

fedml build -t server \
-sf $SOURCE \
-ep $ENTRY \
-cf $CONFIG \
-df $DEST