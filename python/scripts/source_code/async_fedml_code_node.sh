#!/bin/bash
DEV_NODE=

LOCAL_PATH=/Users/hchaoyan/source/FedML/

REMOTE_PATH=/home/ec2-user/FedML

alias ws-sync='rsync -avP -e ssh --exclude '.idea' $LOCAL_PATH $DEV_NODE:$REMOTE_PATH'
ws-sync; fswatch -o $LOCAL_PATH | while read f; do ws-sync; done