wget -r -np -R "index.html*" https://archive.ics.uci.edu/ml/machine-learning-databases/00442/
rsync -a archive.ics.uci.edu/ml/machine-learning-databases/00442/* ./N-BaIoT
rm -r archive.ics.uci.edu
find ./N-BaIoT -name '*.rar' -execdir unar {} \; -exec rm {} \;
python dataset.py