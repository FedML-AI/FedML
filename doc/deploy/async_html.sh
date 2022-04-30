#!/bin/bash
# 8 GPUs/node
FedML_DEV=mobileapi.fedml.ai

LOCAL_PATH=/Users/chaoyanghe/sourcecode/FedML_product/FedML-refactor/doc/en/_build/html/

REMOTE_PATH=/home/ubuntu/doc

PEM_PATH=/Users/chaoyanghe/sourcecode/FedML_product/FedML-refactor/doc/deploy/fedmobile.pem

alias ws-sync='rsync -avL --progress -e "ssh -i $PEM_PATH" $LOCAL_PATH ubuntu@$FedML_DEV:$REMOTE_PATH'
ws-sync; fswatch -o $LOCAL_PATH | while read f; do ws-sync; done