DATA_PATH=~/fedgraphnn_data/sider

wget -N --no-check-certificate --no-proxy -P $DATA_PATH https://fedmol.s3-us-west-1.amazonaws.com/datasets/sider/sider.zip && cd $DATA_PATH &&
unzip sider.zip