DATA_PATH=~/fedgraphnn_data/muv

wget -N --no-check-certificate --no-proxy -P $DATA_PATH https://fedmol.s3-us-west-1.amazonaws.com/datasets/muv/muv.zip && cd $DATA_PATH &&
unzip muv.zip