SOURCE=.
ENTRY=main_fedml_object_detection.py
CONFIG=config
DEST=./mlops
IGNORE=__pycache__,*.git,runs,.vscode,devops,.idea,*.ipynb,cache
fedml build -t client -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST --ignore $IGNORE


SOURCE=.
ENTRY=main_fedml_object_detection.py
CONFIG=config
DEST=./mlops
IGNORE=__pycache__,*.git,runs,.vscode,devops,.idea,*.ipynb,cache
fedml build -t server -sf $SOURCE -ep $ENTRY -cf $CONFIG -df $DEST --ignore $IGNORE
