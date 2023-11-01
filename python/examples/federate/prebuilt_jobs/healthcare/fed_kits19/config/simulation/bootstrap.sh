### don't modify this part ###
set -x
##############################


### please customize your script in this region ####
# To work around the 'sklearn' PyPI package is deprecated, use 'scikit-learn' ERROR.
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install flamby[all_extra]
DATA_PATH=$HOME/healthcare/kits19
mkdir -p $DATA_PATH


### don't modify this part ###
exit 0
##############################