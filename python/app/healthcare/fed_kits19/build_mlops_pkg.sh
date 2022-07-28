SOURCE=.
ENTRY=main_fedml_kits19.py
CONFIG=config
DEST=./mlops
IGNORE=__pycache__,*.git 
fedml build -t client -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST --ignore $IGNORE


SOURCE=.
ENTRY=main_fedml_kits19.py
CONFIG=config
DEST=./mlops
IGNORE=__pycache__,*.git 
fedml build -t client -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST --ignore $IGNORE