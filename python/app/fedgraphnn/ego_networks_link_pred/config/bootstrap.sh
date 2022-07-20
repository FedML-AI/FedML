### don't modify this part ###
set -x
##############################


### please customize your script in this region ####
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html

DATA_PATH=~/fedgraphnn_data/CS/
mkdir DATA_PATH

python ../data/sampleEgonetworks.py --path $DATA_PATH --data CS --ego_number 1000 --hop_number 2


### don't modify this part ###
exit 0
##############################