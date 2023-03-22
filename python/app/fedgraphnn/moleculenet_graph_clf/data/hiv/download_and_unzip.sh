DATA_PATH=~/fedgraphnn_data/hiv

wget -N --no-check-certificate --no-proxy -P $DATA_PATH https://fedmol.s3-us-west-1.amazonaws.com/datasets/hiv/hiv.zip && cd $DATA_PATH  && unzip hiv.zip 