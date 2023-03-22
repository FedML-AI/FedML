import copy
import logging
import os
import pickle
import random

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.utils import to_networkx, degree


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def split_graph(graph, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1
    graph.edge_label = graph.edge_label.float()
    edge_size = graph.edge_label.size()[0]
    train_num = int(edge_size * train_ratio)
    val_num = max(1, int(edge_size * val_ratio))
    test_num = max(1, int(edge_size * test_ratio))
    train_num = edge_size - val_num - test_num

    [train_split, val_split, test_split] = torch.utils.data.random_split(
        range(edge_size), [train_num, val_num, test_num]
    )
    train_split = torch.tensor(train_split)
    val_split = torch.tensor(val_split)
    test_split = torch.tensor(test_split)
    graph.edge_train = graph.edge_index[:, train_split]
    graph.label_train = graph.edge_label[train_split]
    graph.edge_val = graph.edge_index[:, val_split]
    graph.label_val = graph.edge_label[val_split]
    graph.edge_test = graph.edge_index[:, test_split]
    graph.label_test = graph.edge_label[test_split]

    return graph


def combine_subgraphs(graph_orig, graph_add):
    graph_orig_dic = {}
    for i in range(len(graph_orig.index_orig)):
        graph_orig_dic[graph_orig.index_orig[i]] = i

    mapping_add = {}
    node_count = len(graph_orig.index_orig)

    index_orig_new = []

    for i in range(len(graph_add.index_orig)):
        if graph_add.index_orig[i] in graph_orig_dic:
            mapping_add[i] = graph_orig_dic[i]
        else:
            mapping_add[i] = node_count
            index_orig_new.append(graph_add.index_orig[i])
            node_count += 1
    index_final = torch.cat([graph_orig.index_orig, torch.tensor(index_orig_new)])

    edge_index_new = copy.deepcopy(graph_add.edge_index)
    for key in mapping_add:
        edge_index_new[edge_index_new == key] = mapping_add[key]

    edge_index_final = torch.cat([graph_orig.edge_index, edge_index_new], dim=-1)
    edge_label_final = torch.cat([graph_orig.edge_label, graph_add.edge_label])
    combined_graph = Data(edge_index=edge_index_final)

    combined_graph.edge_label = edge_label_final
    combined_graph.index_orig = index_final
    return combined_graph


def _convert_to_nodeDegreeFeatures(graphs):
    graph_infos = []
    maxdegree = 0
    for i, graph in enumerate(graphs):
        g = to_networkx(graph, to_undirected=True)
        gdegree = max(dict(g.degree).values())
        if gdegree > maxdegree:
            maxdegree = gdegree
        graph_infos.append(
            (graph, g.degree, graph.num_nodes)
        )  # (graph, node_degrees, num_nodes)

    new_graphs = []
    for i, tpl in enumerate(graph_infos):
        # idx, x = tpl[0].edge_index[0], tpl[0].x
        idx, x = torch.tensor(range(tpl[0].num_nodes)), tpl[0].x
        deg = degree(idx, tpl[2], dtype=torch.long)
        deg = F.one_hot(deg, num_classes=maxdegree + 1).to(torch.float)

        new_graph = tpl[0].clone()
        new_graph.__setitem__("x", deg)
        new_graphs.append(new_graph)

    return new_graphs


def _subgraphing(g, partion, mapping_item2category):
    nodelist = [[] for i in set(mapping_item2category.keys())]
    for k, v in partion.items():
        for category in v:
            nodelist[category].append(k)

    graphs = []
    for nodes in nodelist:
        if len(nodes) < 2:
            continue
        graph = nx.subgraph(g, nodes)
        graphs.append(from_networkx(graph))
    return graphs


def _read_mapping(path, data, filename):
    mapping = {}
    with open(os.path.join(path, data, filename)) as f:
        for line in f:
            s = line.strip().split()
            mapping[int(s[0])] = int(s[1])

    return mapping


def _build_nxGraph(path, data, filename, mapping_user, mapping_item):
    G = nx.Graph()
    with open(os.path.join(path, data, filename)) as f:
        for line in f:
            s = line.strip().split()
            s = [int(i) for i in s]
            G.add_edge(mapping_user[s[0]], mapping_item[s[1]], edge_label=s[2])
    dic = {}
    for node in G.nodes:
        dic[node] = node
    nx.set_node_attributes(G, dic, "index_orig")
    return G


def combine_category(graphs, category_split):
    combined_graph = []
    for i in range(len(category_split)):
        ls = category_split[i]
        logging.info("combining subgraphs for " + str(i) + " client")
        graph_new = graphs[ls[0]]
        for i in range(1, len(ls)):
            logging.info("combined " + str(i + 1) + " subgraphs")
            graph_new = combine_subgraphs(graph_new, graphs[ls[i]])
        combined_graph.append(graph_new)
    return combined_graph


def get_data_category(args, path, data, load_processed=True):
    """For link prediction."""
    if load_processed:
        if args.client_num_in_total > 1:
            with open(os.path.join(path, data, "subgraphs.pkl"), "rb") as f:
                graphs = pickle.load(f)
        
        else:
            with open(os.path.join(path, data, 'graph.pkl'), "rb") as f:
                graphs = pickle.load(f)

    else:

        logging.info("read mapping")
        mapping_user = _read_mapping(path, data, "user.dict")
        mapping_item = _read_mapping(path, data, "item.dict")
        mapping_item2category = _read_mapping(path, data, "category.dict")
        logging.info("build networkx graph")

        graph = _build_nxGraph(path, data, "graph.txt", mapping_user, mapping_item)
        logging.info("get partion")

        partion = partition_by_category(graph, mapping_item2category)
        logging.info("subgraphing")
        graphs = _subgraphing(graph, partion, mapping_item2category)

    logging.info(
        "check client num is smaller than subgraphs number, which is "
        + str(len(graphs))
    )
    assert args.client_num_in_total <= len(graphs)

    perm = list(torch.randperm(len(graphs)))
    category_split = np.array_split(perm, args.client_num_in_total)
    graphs = combine_category(graphs, category_split)

    logging.info("converting to node degree")
    graphs = _convert_to_nodeDegreeFeatures(graphs)
    graphs_split = []
    logging.info("spliting into trian val and test")
    for g in graphs:
        graphs_split.append(split_graph(g))
    return graphs_split


def partition_by_category(graph, mapping_item2category):
    partition = {}
    for key in mapping_item2category:
        partition[key] = [mapping_item2category[key]]
        for neighbor in graph.neighbors(key):
            if neighbor not in partition:
                partition[neighbor] = []
            partition[neighbor].append(mapping_item2category[key])
    return partition


def create_category_split(
    args, path, data, pred_task="link_prediction", algo="Louvain"
):
    assert pred_task in ["link_prediction"]
    logging.info("reading data")

    graphs_split = get_data_category(args, path, data, load_processed=False)

    return graphs_split


def partition_data_by_category(args, path, compact=True):
    graphs_split = create_category_split(args, path, args.dataset, args.pred_task)

    client_number = len(graphs_split)

    partition_dicts = [None] * client_number

    for client in range(client_number):
        partition_dict = {"graph": graphs_split[client]}
        partition_dicts[client] = partition_dict

    global_data_dict = {
        "graphs": graphs_split,
    }

    return global_data_dict, partition_dicts


# Single process sequential
def load_partition_data(args, path, client_number):
    global_data_dict, partition_dicts = partition_data_by_category(args, path)
    feature_dim = global_data_dict["graphs"][0].x.shape[1]

    data_local_num_dict = {}
    train_data_local_dict = {}

    # This is a PyG Dataloader
    train_data_global = DataLoader(
        global_data_dict["graphs"], batch_size=1, shuffle=True, pin_memory=True
    )

    train_data_num = len(global_data_dict["graphs"])

    for client in range(client_number):
        train_dataset_client = partition_dicts[client]["graph"]

        data_local_num_dict[client] = 1
        train_data_local_dict[client] = DataLoader(
            [train_dataset_client], batch_size=1, shuffle=True, pin_memory=True
        )
        logging.info(
            "Client idx = {}, local sample number = {}".format(
                client, len(train_dataset_client)
            )
        )

    val_data_num = test_data_num = train_data_num
    val_data_local_dict = test_data_local_dict = train_data_local_dict
    val_data_global = test_data_global = train_data_global
    return (
        train_data_num,
        val_data_num,
        test_data_num,
        train_data_global,
        val_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict,
        feature_dim,
    )

def load_recsys_data(args, dataset_name):
    if args.dataset not in ["ciao", "epinions"]:
        raise Exception("no such dataset!")

    args.pred_task = "link_prediction"

    args.metric = "MAE"

    if args.model == "gcn":
        args.normalize_features = True
        args.normalize_adjacency = True

    (
        train_data_num,
        val_data_num,
        test_data_num,
        train_data_global,
        val_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict,
        feature_dim,
    ) = load_partition_data(args, args.data_dir, args.client_num_in_total)

    dataset = [
        train_data_num,
        val_data_num,
        test_data_num,
        train_data_global,
        val_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict,
        feature_dim,
    ]

    return dataset