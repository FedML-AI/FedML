:: ### don't modify this part ###
:: ##############################


:: ### please customize your script in this region ####
set DATA_PATH=%userprofile%\fedml_data
if exist %DATA_PATH% (echo Exist %DATA_PATH%) else mkdir %DATA_PATH%

pip install MNN==1.1.6

:: ### don't modify this part ###
echo [FedML]Bootstrap Finished
:: ##############################