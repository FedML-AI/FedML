#!/usr/bin/env bash

mkdir datasets
cd datasets
wget --no-check-certificate --no-proxy https://fedml.s3-us-west-1.amazonaws.com/shakespeare.tar.bz2
tar -xvf shakespeare.tar.bz2
