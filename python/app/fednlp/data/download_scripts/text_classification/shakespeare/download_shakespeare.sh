mkdir train
cd train

#https://drive.google.com/file/d/1mD6_4ju7n2WFAahMKDtozaGxUASaHAPH/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt \
--keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1mD6_4ju7n2WFAahMKDtozaGxUASaHAPH' \
-O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1mD6_4ju7n2WFAahMKDtozaGxUASaHAPH" -O "all_data_niid_2_keep_0_train_8.json" && rm -rf /tmp/cookies.txt
cd ..

mkdir test
cd test

#https://drive.google.com/file/d/1GERQ9qEJjXk_0FXnw1JbjuGCI-zmmfsk/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt \
--keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GERQ9qEJjXk_0FXnw1JbjuGCI-zmmfsk' \
-O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GERQ9qEJjXk_0FXnw1JbjuGCI-zmmfsk" -O "all_data_niid_2_keep_0_test_8.json" && rm -rf /tmp/cookies.txt
cd ..