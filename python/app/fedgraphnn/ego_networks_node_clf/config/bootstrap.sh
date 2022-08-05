### don't modify this part ###
set -x
##############################


### please customize your script in this region ####
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html

DATA_PATH=~/ego_networks_node_clf/
mkdir $DATA_PATH

python ./fedml_ego_networks_node_clf.py --path ego_networks_node_clf/ --data cora --ego_number 1000 --hop_number 2 --cf config/simulation/fedml_config.yaml


### don't modify this part ###
echo "[FedML]Bootstrap Finished"
##############################