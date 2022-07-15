#CPU installation

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html

BASE_DATA_PATH=~/fedgraphnn_data
DATASET=clintox
wget --no-check-certificate --no-proxy -P $BASE_DATA_PATH/$DATASET https://fedmol.s3-us-west-1.amazonaws.com/datasets/clintox/clintox.zip &&
cd $BASE_DATA_PATH/$DATASET &&
unzip clintox.zip && rm clintox.zip


