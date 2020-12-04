#!/usr/bin/env bash

mkdir datasets
cd datasets

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1oJE5dtifbaDEFGTUjoIwLEtUTfL9XhcU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1oJE5dtifbaDEFGTUjoIwLEtUTfL9XhcU" -O "fed_cifar100_test.h5" && rm -rf /tmp/cookies.txt 
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1B9D5JCWAb2W0BvhLw7mzKgsGnOz7N774' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1B9D5JCWAb2W0BvhLw7mzKgsGnOz7N774" -O "fed_cifar100_train.h5" && rm -rf /tmp/cookies.txt 
