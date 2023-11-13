SOURCE=.
ENTRY=tf_fedml_main.py
CONFIG=config
DEST=./mlops
fedml build -t client -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST 

fedml build -t server -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST