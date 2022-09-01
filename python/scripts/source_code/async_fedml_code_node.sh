#!/bin/bash
# https://www.ibm.com/support/pages/configuring-ssh-login-without-password
DEV_NODE=chaoyanghe@everest.usc.edu

LOCAL_PATH=/Users/chaoyanghe/sourcecode/FedML_product/FedML/

REMOTE_PATH=/home/chaoyanghe/FedML

alias ws-sync='rsync -avP -e ssh --exclude '.idea' $LOCAL_PATH $DEV_NODE:$REMOTE_PATH'
ws-sync; fswatch -o $LOCAL_PATH | while read f; do ws-sync; done