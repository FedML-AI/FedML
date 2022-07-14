conda install pyg -c pyg
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

DATA_PATH=~/fedgraphnn
wget --no-check-certificate --no-proxy -P $DATA_PATH https://fedmol.s3-us-west-1.amazonaws.com/datasets/clintox/clintox.zip &&
cd $DATA_PATH &&
unzip clintox.zip && rm clintox.zip