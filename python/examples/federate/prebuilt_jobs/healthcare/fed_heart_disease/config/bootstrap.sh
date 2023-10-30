### don't modify this part ###
set -x
##############################


### please customize your script in this region ####
pip install batchgenerators
pip install seaborn
git clone https://github.com/owkin/FLamby.git
cd FLamby
python3 setup.py install

DATA_PATH=$HOME/healthcare/heart_disease
mkdir -p $DATA_PATH


### don't modify this part ###
exit 0
##############################