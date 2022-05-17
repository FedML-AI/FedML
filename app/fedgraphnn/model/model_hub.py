import logging
import torch_geometric
from fedgraphnn.model.moleculenet.gat_readout import GatMoleculeNet
from fedgraphnn.model.moleculenet.gcn_readout import GcnMoleculeNet
from fedgraphnn.model.moleculenet.sage_readout import SageMoleculeNet
from fedgraphnn.model.recommender_system.gcn_link import GCNLinkPred
from fedgraphnn.model.recommender_system.gat_link import GATLinkPred
from fedgraphnn.model.recommender_system.sage_link import SAGELinkPred
from fedgraphnn.model.social_networks.gin import GIN
from fedgraphnn.model.subgraph_level.rgcn import RGCNEncoder, DistMultDecoder
from fedgraphnn.model.ego_networks.gcn import GCNNodeCLF
from fedgraphnn.model.ego_networks.sgc import SGCNodeCLF
from fedgraphnn.model.ego_networks.gat import GATNodeCLF
from fedgraphnn.model.ego_networks.sage import SAGENodeCLF

def create(args, model_name, feat_dim, num_cats, output_dim):
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (model_name, output_dim)
    )
    if args.dataset in ["sider", "clintox", "bbbp", "esol", "freesolv", "herg", "lipo", "pcba", "tox21", "toxcast", "muv","hiv" , "qm7" , "qm8" , "qm9"]:
        #MoleculeNet datasets
        if model_name == "graphsage":
            model = SageMoleculeNet(
                feat_dim,
                args.hidden_size,
                args.node_embedding_dim,
                args.dropout,
                args.readout_hidden_dim,
                args.graph_embedding_dim,
                num_cats,
            )
        elif model_name == "gat":
            model = GatMoleculeNet(
                feat_dim,
                args.hidden_size,
                args.node_embedding_dim,
                args.dropout,
                args.alpha,
                args.num_heads,
                args.readout_hidden_dim,
                args.graph_embedding_dim,
                num_cats,
            )
        elif model_name == "gcn":
            model = GcnMoleculeNet(
                feat_dim,
                args.hidden_size,
                args.node_embedding_dim,
                args.dropout,
                args.readout_hidden_dim,
                args.graph_embedding_dim,
                num_cats,
                sparse_adj=args.sparse_adjacency,
            )
        else:
            raise Exception("such model does not exist !")
        logging.info("done")
    elif args.dataset in ["ciao","epinions"]:
        #Recommender Systems
        if model_name == "gcn":
            model = GCNLinkPred(args.feature_dim, args.hidden_size, args.node_embedding_dim)
        elif model_name == "gat":
            model = GATLinkPred(
                args.feature_dim, args.hidden_size, args.node_embedding_dim, args.num_heads
            )
        elif model_name == "sage":
            model = SAGELinkPred(args.feature_dim, args.hidden_size, args.node_embedding_dim)
        else:
            raise Exception("such model does not exist !")
    elif args.dataset in ["COLLAB","REDDIT-BINARY","REDDIT-MULTI-5K","REDDIT-MULTI-12K","IMDB-BINARY","IMDB-MULTI"]:
        #Social-networks
        model = GIN(args.feature_dim, args.hidden_size,num_cats,args.n_layers,args.dropout,args.eps)
    elif args.dataset in ["YAGO3-10", "wn18rr", "FB15k-237"]:
        #Subgraph level
        if model_name == "rgcn":
            model = torch_geometric.nn.GAE(
                RGCNEncoder(34662, hidden_channels=500, num_relations=11),
                DistMultDecoder(11 // 2, hidden_channels=500),
            )
        else:
            raise Exception("such model does not exist !")
    elif args.dataset in ["CS", "Physics", "cora", "citeseer", "DBLP", "PubMed"]:
        #Federated Node classification
        if model_name == "gcn":
            model = GCNNodeCLF(
                nfeat=feat_dim,
                nhid=args.hidden_size,
                nclass=num_cats,
                nlayer=args.n_layers,
                dropout=args.dropout,
            )
        elif model_name == "sgc":
            model = SGCNodeCLF(in_dim=feat_dim, num_classes=num_cats, K=args.n_layers)
        elif model_name == "sage":
            model = SAGENodeCLF(
                nfeat=feat_dim,
                nhid=args.hidden_size,
                nclass=num_cats,
                nlayer=args.n_layers,
                dropout=args.dropout,
            )
        elif model_name == "gat":
            model = GATNodeCLF(
                in_channels = args.feature_dim,
                out_channels = num_cats, 
                dropout=args.dropout,
            )
        else:
            # MORE MODELS
            raise Exception("such model does not exist !")
    else:
        raise Exception("such dataset does not exist !")

    return model