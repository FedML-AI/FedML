#!/usr/bin/env bash

mkdir datasets
cd datasets
wget --no-check-certificate --no-proxy https://fedml.s3-us-west-1.amazonaws.com/fed_cifar100.tar.bz2
tar -xvf fed_cifar100.tar.bz2
