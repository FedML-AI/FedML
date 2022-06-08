# Scripts to start the FL training

1. download the dataset
```
cd data/esol
sh download_and_unzip.sh
```

2. install PyG
Linux/MacOS (x86):
```
pip install matplotli
```

MacOS M1 chip:
```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
```
and install other packages
```
pip install -r requirements.txt
```

3. start the training
```
sh run_moleculenet_reg.sh 4
```
