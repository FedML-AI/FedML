# Scripts to start the FL training

1. download the dataset and generate partitions
```
cd data/YAGO3-10
sh download_and_unzip_YAGO3-10.sh
cd ../data/wn18rr
sh download_and_unzip_wn18rr.sh
cd ../data/FB15K-237
sh download_and_unzip_FB15K-237.sh
cd ..
python partitionKG.py --data FB15k-237/ --pred_task link --path ./
python partitionKG.py --data YAGO3-10/ --pred_task link --path ./
python partitionKG.py --data FB15K-237/ --pred_task link --path ./

```

2. start the training
```
sh run_fed_link_pred.sh 4
```