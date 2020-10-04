#!/usr/bin/env bash

mkdir datasets
cd datasets

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zpk83oW6f9YRjesewfq4l75Ubx6z3t_A' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zpk83oW6f9YRjesewfq4l75Ubx6z3t_A" -O "stackoverflow_test.h5" && rm -rf /tmp/cookies.txt 
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1JvSSER1cnXq2Q-lNdLO-K5S2pyN0Glx7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1JvSSER1cnXq2Q-lNdLO-K5S2pyN0Glx7" -O "stackoverflow_train.h5" && rm -rf /tmp/cookies.txt 
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RvcjTkJrEwR51m1fJpI_G9ADTVSXYfQW' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1RvcjTkJrEwR51m1fJpI_G9ADTVSXYfQW" -O "stackoverflow_held_out.h5" && rm -rf /tmp/cookies.txt 

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16Xp1jVx7aSS319P6TzTgnWoQD0g9-mwZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16Xp1jVx7aSS319P6TzTgnWoQD0g9-mwZ" -O "stackoverflow.tag_count" && rm -rf /tmp/cookies.txt 
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ubwwH7WHFRY7mUKSo28q8npJhdi2HWG6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ubwwH7WHFRY7mUKSo28q8npJhdi2HWG6" -O "stackoverflow.word_count" && rm -rf /tmp/cookies.txt 
