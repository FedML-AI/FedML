# Scripts to start the FL training

1. download the dataset
```
cd data/esol
sh download_and_unzip.sh
```

2. install PyG
```
conda install pyg -c pyg
```

3. start the training
```
sh run_moleculenet_reg.sh 4
```
