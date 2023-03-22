DATA_PATH=~/fedgraphnn_data/qm9

wget -N --no-check-certificate --no-proxy -P $DATA_PATH https://fedmol.s3-us-west-1.amazonaws.com/datasets/qm9/qm9.zip && cd $DATA_PATH &&
unzip qm9.zip