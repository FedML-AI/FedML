# This US server is too slow for US users.
#wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz

# We move the file to Google Driver for faster downloading
#https://drive.google.com/file/d/1WimEOXYdCdtry4cZQrJl3DzrAKcEJuyA/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt \
--keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WimEOXYdCdtry4cZQrJl3DzrAKcEJuyA' \
-O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WimEOXYdCdtry4cZQrJl3DzrAKcEJuyA" -O "CINIC-10.tar.gz" && rm -rf /tmp/cookies.txt

tar xvzf CINIC-10.tar.gz