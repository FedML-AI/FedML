#!/usr/bin/env bash
set -x

WORKSPACE=/home/ec2-user/FedML

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

USER=fedml
HOME=/home/$USER


cd $WORKSPACE/experiments/distributed/

wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
wandb on
sleep 100000000
