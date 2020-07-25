#!/usr/bin/env bash

NAME="shakespeare"

cd ../utils

python3 stats.py --name $NAME

cd ../$NAME