# Ego Networks - Link Prediction

## Motivation

Link prediction, related to the likelihood of having a link between two nodes of the network that are not connected, is a key problem in social network analysis. It is important in the node-level FL setting, where friend
suggestion and social relation profiling can be attempted in users’ ego-networks, for example In federated settings, it is possible to represent each user in a graph as a ego network as each user’s personal data can be sensitive and only
visible to his/her k-hop neighbors. Thus, it is natural to consider node-level FL in social networks with clients holding the user ego-networks. To simulate this scenario, we use the open social networks
and publication networks and partition them into sets of ego-networks. For the rest of the tutorial, we refer to our [source code 

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

Arguments for data preparation can be found in our source coede.

## Training

```
WORKSPACE=./FedML/app/fedgraphnn/app/ego_networks_link_pred
cd $WORKSPACE

sh run_fed_link_pred.sh 4
```
