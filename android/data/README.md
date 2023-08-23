# Dataset Preparation
```
pip install idx2numpy
python data_split.py
```
# Dataset Installation
For the MNIST example, please put the data files `MNIST/raw/client0` to device 0, `MNIST/raw/client0` to device 1, and so on.

After connecting your Android smartphone to the laptop, we can call following script to install dataset for device 0~3:

```
bash prepare.sh 0
bash prepare.sh 1
bash prepare.sh 2
bash prepare.sh 3
```

Double check whether the installation is correct:
```
adb shell
cd /storage/emulated/0/Android/data/ai.fedml.edgedemo/files/dataset
```