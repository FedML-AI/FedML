DATA_PATH=~/fedgraphnn_data/esol

wget -N --no-check-certificate --no-proxy -P $DATA_PATH https://fedmol.s3-us-west-1.amazonaws.com/datasets/esol/esol.zip && cd $DATA_PATH &&
unzip esol.zip