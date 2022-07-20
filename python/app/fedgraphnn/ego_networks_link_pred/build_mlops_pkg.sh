SOURCE=.
ENTRY=fedml_ego_networks_link_pred.py
CONFIG=config
DEST=./mlops
fedml build -t client -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST


SOURCE=.
ENTRY=fedml_ego_networks_link_pred.py
CONFIG=config
DEST=./mlops
fedml build -t server -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST