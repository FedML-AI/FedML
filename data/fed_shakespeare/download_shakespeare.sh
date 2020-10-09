#!/usr/bin/env bash

mkdir datasets
cd datasets

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1OT1pu0SEHyuJxS7RRqe9me2GiutBzcJM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1OT1pu0SEHyuJxS7RRqe9me2GiutBzcJM" -O "shakespeare_test.h5" && rm -rf /tmp/cookies.txt 
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1HwmPKCx4SUnkols-ubv8ugKtorMN6r86' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1HwmPKCx4SUnkols-ubv8ugKtorMN6r86" -O "shakespeare_train.h5" && rm -rf /tmp/cookies.txt 
