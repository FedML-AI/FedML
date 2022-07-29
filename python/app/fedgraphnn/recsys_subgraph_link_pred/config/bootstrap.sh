### don't modify this part ###
set -x
##############################


### please customize your script in this region ####
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html

DATA_PATH=~/fedgraphnn_data/ciao/
mkdir $DATA_PATH

cp -R ../data/ciao/ $DATA_PATH
### don't modify this part ###
exit 0
##############################