#!/usr/bin/env bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yxkLjbrONC8JnAjXHW9HBcVRA48nIL9d' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yxkLjbrONC8JnAjXHW9HBcVRA48nIL9d" -O "cifar100_test.h5" && rm -rf /tmp/cookies.txt 
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1jVU3HIEYooZwnl2cO8tausbaMrTSlsc9' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1jVU3HIEYooZwnl2cO8tausbaMrTSlsc9" -O "cifar100_train.h5" && rm -rf /tmp/cookies.txt 

