DATA_PATH=~/fedgraphnn_data/herg

wget -N --no-check-certificate --no-proxy -P $DATA_PATH https://fedmol.s3-us-west-1.amazonaws.com/datasets/herg/herg.zip && cd $DATA_PATH &&
unzip herg.zip