SOURCE=./
ENTRY=torch_server.py
CONFIG=./config
DEST=./mlops
fedml build -t server -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST