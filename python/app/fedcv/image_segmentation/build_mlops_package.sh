SOURCE=.
ENTRY=main_fedml_image_segmentation.py
CONFIG=config
DEST=./mlops
fedml build -t client -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST


SOURCE=.
ENTRY=main_fedml_image_segmentation.py
CONFIG=config
DEST=./mlops
fedml build -t server -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST
