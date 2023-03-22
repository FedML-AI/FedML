DATA_PATH=~/fedgraphnn_data/freesolv

wget -N --no-check-certificate --no-proxy -P $DATA_PATH https://fedmol.s3-us-west-1.amazonaws.com/datasets/freesolv/freesolv.zip && cd $DATA_PATH &&
unzip freesolv.zip