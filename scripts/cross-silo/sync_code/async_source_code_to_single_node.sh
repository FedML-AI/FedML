#!/bin/bash
DEV_NODE=$1
LOCAL_PATH=/Users/hchaoyan/source/FedML/
REMOTE_PATH=/home/ec2-user/FedML
rsync -avP -e ssh --exclude '.idea' $LOCAL_PATH $DEV_NODE:$REMOTE_PATH