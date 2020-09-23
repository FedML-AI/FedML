#!/usr/bin/env bash

if [[ -n "$1" ]] && [[ "${1#*.}" == "digit" ]]; then 
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1mhKQN1LjaweRU1iidnfvWoj2igEY8szt' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1mhKQN1LjaweRU1iidnfvWoj2igEY8szt" -O "emnist_test_only_digits.h5" && rm -rf /tmp/cookies.txt 
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-Kv88Gyoqnyn_LBdMN-8sH4Eedp09V8o' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-Kv88Gyoqnyn_LBdMN-8sH4Eedp09V8o" -O "emnist_train_only_digits.h5" && rm -rf /tmp/cookies.txt 
else
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wUzroOnlArzcFe8MT-PKI1jgubfLuTfd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wUzroOnlArzcFe8MT-PKI1jgubfLuTfd" -O "emnist_test.h5" && rm -rf /tmp/cookies.txt 
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1FS8Qev79yLRY7jmtTcv5ItJlXFnie469' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1FS8Qev79yLRY7jmtTcv5ItJlXFnie469" -O "emnist_train.h5" && rm -rf /tmp/cookies.txt 
fi



