import os
import argparse
import random
import networkx as nx
import copy
import logging
import pickle
import pandas as pd
import community as community_louvain

import torch
from torch_geometric.utils import from_networkx

from pathlib import Path
import time


def _read_mapping(path, data, filename):
    mapping = dict()
    df = pd.read_csv(
        os.path.join(path, data, filename), sep="\t", header=None, index_col=None
    )
    for _, row in df.iterrows():
        mapping[row[1]] = int(row[0])

    return mapping


def _build_pygGraph(relType, df, mapping_entities, mapping_relations):
    df[0].replace(mapping_entities, inplace=True)
    df[1].replace(mapping_relations, inplace=True)
    df[2].replace(mapping_entities, inplace=True)

    g = nx.Graph()
    g.add_edges_from(zip(df[0], df[2]), edge_label=mapping_relations[relType])

    gObj = from_networkx(g)
    gObj.number_of_nodes = g.number_of_nodes()

    return gObj


def _build_graphs_by_relType(path, data, filename, mapping_entities, mapping_relations):
    df = pd.read_csv(
        os.path.join(path, data, filename), sep="\t", header=None, index_col=None
    )

    graphs = [
        _build_pygGraph(relType, group, mapping_entities, mapping_relations)
        for relType, group in df.groupby(1)
    ]
    return graphs


def get_data_community_byRelType(path, data):
    """For link prediction with communities grouped by the relation type."""

    mapping_entities = _read_mapping(path, data, "entities.dict")
    mapping_relations = _read_mapping(path, data, "relations.dict")

    start = time.time()
    graphs_train = _build_graphs_by_relType(
        path, data, "train.txt", mapping_entities, mapping_relations
    )
    print("built train graphs:", time.time() - start)
    start = time.time()
    graphs_val = _build_graphs_by_relType(
        path, data, "valid.txt", mapping_entities, mapping_relations
    )
    print("built val graphs:", time.time() - start)
    start = time.time()
    graphs_test = _build_graphs_by_relType(
        path, data, "test.txt", mapping_entities, mapping_relations
    )
    print("built test graphs:", time.time() - start)

    # number of graphs == number of relation type

    return graphs_train, graphs_val, graphs_test


def _subgraphing(g, partion):
    nodelist = [None] * len(set(partion.values()))
    for k, v in partion.items():
        if nodelist[v] is None:
            nodelist[v] = []
        nodelist[v].append(k)

    graphs = []
    for nodes in nodelist:
        if len(nodes) < 2:
            continue
        gObj = from_networkx(nx.subgraph(g, nodes))
        gObj.number_of_nodes = len(nodes)
        graphs.append(gObj)
    return graphs


def _read_mapping(path, data, filename):
    mapping = dict()
    df = pd.read_csv(
        os.path.join(path, data, filename), sep="\t", header=None, index_col=None
    )
    for _, row in df.iterrows():
        mapping[row[1]] = int(row[0])

    return mapping


def _build_nxGraph(path, data, filename, mapping_entities, mapping_relations):
    G = nx.Graph()
    df = pd.read_csv(
        os.path.join(path, data, filename), sep="\t", header=None, index_col=None
    )
    for _, row in df.iterrows():
        G.add_edge(
            mapping_entities[row[0]],
            mapping_entities[row[2]],
            edge_label=mapping_relations[row[1]],
        )

    return G


def get_data_community(path, data, algo="Louvain"):
    """For relation type prediction."""

    mapping_entities = _read_mapping(path, data, "entities.dict")
    mapping_relations = _read_mapping(path, data, "relations.dict")

    g_train = _build_nxGraph(
        path, data, "train.txt", mapping_entities, mapping_relations
    )
    g_val = _build_nxGraph(path, data, "valid.txt", mapping_entities, mapping_relations)
    g_test = _build_nxGraph(path, data, "test.txt", mapping_entities, mapping_relations)

    assert algo in [
        "Louvain",
        "girvan_newman",
        "Clauset-Newman-Moore",
        "asyn_lpa_communities",
        "label_propagation_communities",
    ]

    start = time.time()
    if algo == "Louvain":
        partion = community_louvain.best_partition(g_train)
        graphs_train = _subgraphing(g_train, partion)
        partion = community_louvain.best_partition(g_val)
        graphs_val = _subgraphing(g_val, partion)
        partion = community_louvain.best_partition(g_test)
        graphs_test = _subgraphing(g_test, partion)
    else:
        raise NotImplementedError
    print(f"Partitioned by {algo}:", time.time() - start)

    # algorithms:
    # Louvain
    # girvan_newman
    # greedy_modularity_communities
    # asyn_lpa_communities
    # label_propagation_communities

    return graphs_train, graphs_val, graphs_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", help="the path for loading dataset", type=str, default="./data"
    )
    parser.add_argument("--data", help="the name of dataset", type=str)
    parser.add_argument(
        "--pred_task", help='the prediction task: ["relation", "link"]', type=str
    )
    parser.add_argument(
        "--algo", help="the community detect method", type=str, default="Louvain"
    )

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    if args.pred_task == "relation":
        graphs_train, graphs_val, graphs_test = get_data_community(
            args.path, args.data, args.algo
        )
        outpath = os.path.join(path, data, "subgraphs_byLouvain")

    if args.pred_task == "link":
        graphs_train, graphs_val, graphs_test = get_data_community_byRelType(
            args.path, args.data
        )
        outpath = os.path.join(path, data, "subgraphs_byRelType")

    Path(outpath).mkdir(parents=True, exist_ok=True)
    pickle.dump(graphs_train, open(os.path.join(outpath, "train.pkl"), "wb"))
    print(f"Wrote to {os.path.join(outpath, 'train.pkl')}.")
    pickle.dump(graphs_val, open(os.path.join(outpath, "valid.pkl"), "wb"))
    print(f"Wrote to {os.path.join(outpath, 'valid.pkl')}.")
    pickle.dump(graphs_test, open(os.path.join(outpath, "test.pkl"), "wb"))
    print(f"Wrote to {os.path.join(outpath, 'test.pkl')}.")
