SOURCE=.
ENTRY=main_fedml_chestxray_clf.py
CONFIG=config
DEST=./mlops
fedml build -t client -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST


SOURCE=.
ENTRY=main_fedml_chestxray_clf.py
CONFIG=config
DEST=./mlops
fedml build -t server -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST
