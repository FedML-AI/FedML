DATA_PATH=~/fedgraphnn_data/bbbp

wget -N --no-check-certificate --no-proxy -P $DATA_PATH https://fedmol.s3-us-west-1.amazonaws.com/datasets/bbbp/bbbp.zip && cd $DATA_PATH &&
unzip bbbp.zip