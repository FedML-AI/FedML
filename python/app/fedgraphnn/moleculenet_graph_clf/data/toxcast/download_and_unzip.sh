DATA_PATH=~/fedgraphnn_data/toxcast

wget -N --no-check-certificate --no-proxy -P $DATA_PATH https://fedmol.s3-us-west-1.amazonaws.com/datasets/toxcast/toxcast.zip && cd $DATA_PATH &&
unzip toxcast.zip