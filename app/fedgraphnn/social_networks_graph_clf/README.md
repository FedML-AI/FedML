# Scripts to start the FL training

1. download the dataset
```
cd data/esol
sh download_and_unzip.sh
```

2. install PyG
Linux/MacOS (x86):
```
conda install pyg -c pyg
```

MacOS M1 chip:
```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
```

3. start the training
```
sh run_moleculenet_reg.sh 4
```
