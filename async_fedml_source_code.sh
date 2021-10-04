#!/bin/bash
# 8 GPUs/node
FedML_DEV=everest.usc.edu

LOCAL_PATH=/Users/chaoyanghe/sourcecode/FedML/

REMOTE_PATH=/home/chaoyanghe/FedML

alias ws-sync='rsync -avP -e ssh --exclude '.idea' $LOCAL_PATH chaoyanghe@$FedML_DEV:$REMOTE_PATH'
ws-sync; fswatch -o $LOCAL_PATH | while read f; do ws-sync; done