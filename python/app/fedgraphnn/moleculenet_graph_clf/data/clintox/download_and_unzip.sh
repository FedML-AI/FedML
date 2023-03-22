DATA_PATH=~/fedgraphnn_data/clintox

wget -N --no-check-certificate --no-proxy -P $DATA_PATH https://fedmol.s3-us-west-1.amazonaws.com/datasets/clintox/clintox.zip && cd $DATA_PATH &&
unzip clintox.zip