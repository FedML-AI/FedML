#!/bin/bash

echo "bootstrap out"
mkdir -p ~/fednlp_data

pip3 install -r ./requirements.txt

bash ./download_data.sh
bash ./download_partition.sh

exit 0

# pip install fedml==0.7.15
#pip install --upgrade fedml
