import logging

from fedgraphnn.model.moleculenet.gat_readout import GatMoleculeNet
from fedgraphnn.model.moleculenet.gcn_readout import GcnMoleculeNet
from fedgraphnn.model.moleculenet.sage_readout import SageMoleculeNet


def create(args, model_name, feat_dim, num_cats, output_dim):
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (model_name, output_dim)
    )
    if args.dataset in ["SIDER", "ClinTox", "BBPB", "BACE", "PCBA", "Tox21","MUV","HIV"]:
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
            #trainer = SageMoleculeNetTrainer(model)
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
            #trainer = GatMoleculeNetTrainer(model)
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
            #trainer = GcnMoleculeNetTrainer(model)
        else:
            raise Exception("such model does not exist !")
        logging.info("done")
    elif args.dataset 
    return model