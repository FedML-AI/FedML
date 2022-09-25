:: ### don't modify this part ###
:: ##############################


:: ### please customize your script in this region ####
pip install opencv-python-headless pandas matplotlib seaborn addict
set DATA_PATH=%userprofile%\fedcv_data
if exist %DATA_PATH% (echo Exist %DATA_PATH%) else mkdir %DATA_PATH%


:: ### don't modify this part ###
echo [FedML]Bootstrap Finished
:: ##############################