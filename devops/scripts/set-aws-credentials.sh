#!/bin/bash

aws_access_key_id=$1
aws_secret_access_key=$2
region=$3

mkdir -p /root/.aws
echo "[default]" > /root/.aws/credentials
echo "aws_access_key_id = ${aws_access_key_id}" >> /root/.aws/credentials
echo "aws_secret_access_key = ${aws_secret_access_key}" >> /root/.aws/credentials
echo "[default]" > /root/.aws/config
echo "region = ${region}" >> /root/.aws/config



