DATA_PATH=~/fedgraphnn_data/bace

wget -N --no-check-certificate --no-proxy -P $DATA_PATH https://fedmol.s3-us-west-1.amazonaws.com/datasets/bace/bace.zip && cd $DATA_PATH &&
unzip bace.zip