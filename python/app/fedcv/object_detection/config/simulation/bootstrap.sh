### don't modify this part ###
set -x
##############################


### please customize your script in this region ####
pip install opencv-python-headless pandas matplotlib seaborn addict
DATA_PATH=$HOME/fedcv_data
mkdir -p $DATA_PATH
sh ./../../data/coco128/download_coco128.sh


### don't modify this part ###
echo "[FedML]Bootstrap Finished"
##############################