# Subgraph-Level Link Prediction for Recommender Systems

## Motivation 

As graphs are getting bigger and bigger nowadays, it is common to see their subgraphs
separately collected and stored in multiple local systems. Therefore, it is natural to
consider the subgraph federated learning setting, where each local user/system holds
a small subgraph that may be biased from the distribution of the whole graph.
Hence, the subgraph federated learning aims to collaboratively train a powerful
and generalizable graph mining model without directly sharing their graph data. The first realistic scenario is subgraph link prediction task for recommendation
systems, where the users can interact with items owned by different shops or sectors, which
makes each data owner only holding a part of the global user-item graph. To simulate such
scenarios, we use recommendation datasets from publicly available sources  which have high-quality meta-data information.

## Training
Before starting training, make sure that setup with  `config/fedml_config.yaml` is correct. For this tasks we have two available datasets:  `ciao` and `epinions`. Then, run the following script with the desired number of GPU workers.
```
sh run_fed_subgraph_link_pred.sh 4
```