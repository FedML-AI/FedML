# Scripts to start the FL training

1. download the dataset and generate partitions
```

cd data/YAGO3-10
sh download_and_unzip_YAGO3-10.sh
cd ..

cd ../data/wn18rr
sh download_and_unzip_wn18rr.sh
cd ..

cd ../data/FB15K-237
sh download_and_unzip_FB15K-237.sh
cd ..

pip install python-louvain
python partitionKG.py --data FB15k-237 --pred_task relation --path ./
python partitionKG.py --data YAGO3-10 --pred_task relation --path ./
python partitionKG.py --data wn18rr --pred_task relation --path ./

```

2. start the training
```
sh run_fed_rel_pred.sh 4
```