import fedml
from data.data_loader import *
from model.gcn_link import GCNLinkPred
from model.gat_link import GATLinkPred
from model.sage_link import SAGELinkPred

from trainer.fed_subgraph_lp_trainer import FedSubgraphLPTrainer
from fedml.simulation import SimulatorMPI


def load_data(args):
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
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        feature_dim,
    ]

    return dataset 


def create_model(args, model_name, feature_dim):
    logging.info("create_model. model_name = %s" % (model_name))
    if model_name == "gcn":
        model = GCNLinkPred(feature_dim, args.hidden_size, args.node_embedding_dim)
    elif model_name == "gat":
        model = GATLinkPred(
            feature_dim, args.hidden_size, args.node_embedding_dim, args.num_heads
        )
    elif model_name == "sage":
        model = SAGELinkPred(feature_dim, args.hidden_size, args.node_embedding_dim)
    else:
        raise Exception("such model does not exist !")
    trainer = FedSubgraphLPTrainer(model)
    logging.info("Model and Trainer  - done")
    return model, 

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args)

    # load model
    model, trainer = create_model(args, args.model_name)

    # start training
    simulator = SimulatorMPI(args, device, dataset, model, trainer)
    simulator.run()