:: YOLOv5 🚀 by Ultralytics, GPL-3.0 license
:: Download COCO128 dataset https://www.kaggle.com/ultralytics/coco128 (first 128 images from COCO train2017)
:: Example usage: bash data/scripts/get_coco128.sh
:: parent
:: ├── yolov5
:: └── datasets
::      └── coco128  ← downloads here

:: Download/unzip images and labels
:: unzip directory
set data_dir=~/fedcv_data
if exist %d% (echo Exist %data_dir%) else mkdir %data_dir%

set url=https://github.com/ultralytics/yolov5/releases/download/v1.0/
:: or 'coco128-segments.zip', 68 MB
set f=coco128.zip

echo 'Downloading' %url%%f% ' ...'

curl -L %url%%f% -o %f% && powershell Expand-Archive -LiteralPath %f%  -DestinationPath %data_dir%
:: unzip -o -q %f% -d %data_dir% && rm %f% &

