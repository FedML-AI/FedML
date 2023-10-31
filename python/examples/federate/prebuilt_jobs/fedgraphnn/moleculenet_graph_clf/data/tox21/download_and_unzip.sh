DATA_PATH=~/fedgraphnn_data/tox21

wget -N --no-check-certificate --no-proxy -P $DATA_PATH https://fedmol.s3-us-west-1.amazonaws.com/datasets/tox21/tox21.zip && cd $DATA_PATH &&
unzip tox21.zip