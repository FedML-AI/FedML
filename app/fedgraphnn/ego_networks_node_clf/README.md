# Node-Level Tasks (Ego Networks)


## Data Preparation


For each dataset, ego-networks needs to be sampled first.  
```
WORKSPACE=/home/$USER/FedGraphNN
# WORKSPACE=/Users/chaoyanghe/sourcecode/FedGraphNN
cd $WORKSPACE/data_preprocessing/ego_networks

mkdir $WORKSPACE/data/ego-networks/
mkdir $WORKSPACE/data/ego-networks/cora
mkdir $WORKSPACE/data/ego-networks/citeseer
mkdir $WORKSPACE/data/ego-networks/DBLP
mkdir $WORKSPACE/data/ego-networks/PubMed
mkdir $WORKSPACE/data/ego-networks/CS
mkdir $WORKSPACE/data/ego-networks/Physics

python sampleEgonetworks.py --path ./data/ego-networks/ --data cora --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ./data/ego-networks/ --data citeseer --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ./data/ego-networks/ --data DBLP --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ./data/ego-networks/ --data PubMed --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ./data/ego-networks/ --data CS --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ./data/ego-networks/ --data Physics --ego_number 1000 --hop_number 2
```

#### Arguments for Data Preparation code
This is an ordered list of arguments used in distributed/federated experiments. Note, there are additional parameters for this setting.
```
--path -> the path for loading dataset

--data -> the name of dataset: "CS", "Physics", "cora", "citeseer", "DBLP", "PubMed"

--type_network -> 'the type of dataset': citation", "coauthor"

--ego_number --> 'the number of egos sampled'

--hop_number --> 'the number of hops'
```

#### Datasets to Preprocess

citation networks (# nodes): e.g. DBLP (17716), Cora (2708), CiteSeer (3327), PubMed (19717)

collaboration networks (# nodes): e.g. CS (18333), Physics (34493)
 
 social networks (# ego-networks): e.g. COLLAB, IMDB, DEEZER_EGO_NETS (9629), TWITCH_EGOS (127094)


