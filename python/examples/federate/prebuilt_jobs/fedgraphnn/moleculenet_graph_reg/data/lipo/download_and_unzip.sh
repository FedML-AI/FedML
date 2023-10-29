DATA_PATH=~/fedgraphnn_data/lipo

wget -N --no-check-certificate --no-proxy -P $DATA_PATH https://fedmol.s3-us-west-1.amazonaws.com/datasets/lipo/lipo.zip && cd $DATA_PATH &&
unzip lipo.zip