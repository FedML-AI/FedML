### don't modify this part ###
set -x
##############################


### please customize your script in this region ####
pip install batchgenerators
git clone https://github.com/owkin/FLamby.git
cd FLamby
pip install -e .

DATA_PATH=$HOME/healthcare/heart_disease
mkdir -p $DATA_PATH


### don't modify this part ###
exit 0
##############################