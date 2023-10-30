DATA_PATH=~/fedgraphnn_data/pcba

wget -N --no-check-certificate --no-proxy -P $DATA_PATH https://fedmol.s3-us-west-1.amazonaws.com/datasets/pcba/pcba.zip && cd $DATA_PATH &&
unzip pcba.zip