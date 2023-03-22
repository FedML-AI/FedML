rm -r cnn
rm -r dailymail
rm cnn_stories.tgz
rm dailymail_stories.tgz

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ" -O "cnn_stories.tgz" && rm -rf /tmp/cookies.txt
tar -xvzf cnn_stories.tgz

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs" -O "dailymail_stories.tgz" && rm -rf /tmp/cookies.txt
tar -xvzf dailymail_stories.tgz

rm cnn_stories.tgz
rm dailymail_stories.tgz