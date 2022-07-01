# Ego Networks - Link Prediction

## Motivation

Link prediction, related to the likelihood of having a link between two nodes of the network that are not connected, is a key problem in social network analysis. Many methods have been proposed to solve the problem.
## Data Preparation

For each dataset, ego-networks needs to be sampled first.  
```
cd data/ego-networks

mkdir cora
mkdir citeseer
mkdir DBLP
mkdir PubMed
mkdir CS
mkdir Physics

cd ..
python sampleEgonetworks.py --path ego-networks/ --data cora --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ego-networks/ --data citeseer --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ego-networks/ --data DBLP --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ego-networks/ --data PubMed --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ego-networks/ --data CS --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ego-networks/ --data Physics --ego_number 1000 --hop_number 2
```

#### Arguments for Data Preparation code
This is an ordered list of arguments. Note, there are additional parameters for this setting.
```
--path -> the path for loading dataset

--data -> the name of dataset: "CS", "Physics", "cora", "citeseer", "DBLP", "PubMed"

--type_network -> 'the type of dataset': citation", "coauthor"

--ego_number --> 'the number of egos sampled'

--hop_number --> 'the number of hops'
```

#### Datasets to Preprocess

citation networks (# nodes): DBLP (17716), Cora (2708), CiteSeer (3327), PubMed (19717)

collaboration networks (# nodes): CS (18333), Physics (34493)
 
social networks (# ego-networks): COLLAB, IMDB


## Training

```
WORKSPACE=./FedML/app/fedgraphnn/app/ego_networks_link_pred
cd $WORKSPACE

sh run_fed_link_pred.sh 4
```
