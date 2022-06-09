#!/usr/bin/env bash

USERNAME=$1
PASSWORD=$2

POST_DATA="username=$USERNAME&password=$PASSWORD&submit=Login"

echo $POST_DATA

echo "Logging in using the credentials..."
wget --keep-session-cookies --save-cookies=cookies.txt --post-data $POST_DATA https://www.cityscapes-dataset.com/login/
rm index.html

echo "Downloading gtFine_trainvaltest.zip..."
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
echo "Extracting gtFine_trainvaltest.zip..."
unzip gtFine_trainvaltest.zip
rm gtFine_trainvaltest.zip
rm README*
rm license.txt

echo "Downloading gtCoarse.zip..."
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=2
echo "Extracting gtCoarse.zip..."
unzip gtCoarse.zip
rm gtCoarse.zip
rm README*
rm license.txt

echo "Downloading leftImg8bit_trainvaltest.zip"
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
echo "Extracting leftImg8bit_trainvaltest.zip"
unzip leftImg8bit_trainvaltest.zip
rm leftImg8bit_trainvaltest.zip
rm README*
rm license.txt

echo "Downloading leftImg8bit_trainextra.zip"
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=4
echo "Extracting leftImg8bit_trainextra.zip"
unzip leftImg8bit_trainextra.zip
rm leftImg8bit_trainextra.zip
rm README*
rm license.txt

rm cookies.txt

python process_targets.py
