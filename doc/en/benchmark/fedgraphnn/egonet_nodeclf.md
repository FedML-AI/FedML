# Federated Node Classification over Ego Networks

## Motivation

Node classification on graphs has attracted significant attention as the importance of large-scale
graphs analysis increases in various domains such as bioinformatics and commercial graphs.
For example, in retail services, acquiring the qualitative node representations for items or customers
is critical for improving the quality of recommendation systems. In federated settings, it is possible to represent each user in a graph as a ego network as each user’s personal data can be sensitive and only
visible to his/her k-hop neighbors. Thus, it is natural to consider node-level FL in social networks with clients holding the user ego-networks. To simulate this scenario, we use the open social networks
and publication networks and partition them into sets of ego-networks.

## Data Preparation

For each dataset, ego-networks needs to be sampled first.  To sample, directories have to be created first and then `sampleEgonetworks.py` has to be used to generate ego-networks.

```

mkdir cora
mkdir citeseer
mkdir DBLP
mkdir PubMed
mkdir CS
mkdir Physics

python sampleEgonetworks.py --path ego-networks/ --data cora --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ego-networks/ --data citeseer --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ego-networks/ --data DBLP --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ego-networks/ --data PubMed --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ego-networks/ --data CS --ego_number 1000 --hop_number 2
python sampleEgonetworks.py --path ego-networks/ --data Physics --ego_number 1000 --hop_number 2
```

Arguments for the data preparation process

## Training

```
WORKSPACE=./FedML/app/fedgraphnn/app/ego_networks_node_clf
cd $WORKSPACE

sh run_fed_node_clf.sh 4
```