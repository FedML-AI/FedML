# Scripts to start the FL training

1. download the dataset
```
cd data/sider
sh download_and_unzip.sh
```

2. start the training
```
WORKSPACE=./FedML/app/fedgraphnn/applications/moleculenet_graph_clf
cd $WORKSPACE

sh run_moleculenet_clf.sh 4
```