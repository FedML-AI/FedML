import fedml
from data.data_loader import *
from fedml import FedMLRunner
from model.gat import GATNodeCLF
from model.gcn import GCNNodeCLF
from model.sage import SAGENodeCLF
from model.sgc import SGCNodeCLF
from trainer.federated_nc_trainer import FedNodeClfTrainer


def load_data(args):
    num_cats, feat_dim = 0, 0
    if args.dataset not in ["CS", "Physics", "cora", "citeseer", "DBLP", "PubMed"]:
        raise Exception("no such dataset!")
    elif args.dataset in ["CS", "Physics"]:
        args.type_network = "coauthor"
    else:
        args.type_network = "citation"

    compact = args.model == "graphsage"

    unif = True if args.partition_method == "homo" else False

    if args.model == "gcn":
        args.normalize_features = True
        args.normalize_adjacency = True

    _, _, feat_dim, num_cats = get_data(args.data_cache_dir, args.dataset)

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
    ) = load_partition_data(
        args,
        args.data_cache_dir,
        args.client_num_in_total,
        uniform=unif,
        compact=compact,
        normalize_features=args.normalize_features,
        normalize_adj=args.normalize_adjacency,
    )

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        num_cats,
    ]

    return dataset, num_cats, feat_dim


def create_model(args, feat_dim, num_cats, output_dim=None):
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (args.model, num_cats)
    )
    if args.model == "gcn":
        model = GCNNodeCLF(
            nfeat=feat_dim,
            nhid=args.hidden_size,
            nclass=num_cats,
            nlayer=args.n_layers,
            dropout=args.dropout,
        )
    elif args.model == "sgc":
        model = SGCNodeCLF(in_dim=feat_dim, num_classes=num_cats, K=args.n_layers)
    elif args.model == "sage":
        model = SAGENodeCLF(
            nfeat=feat_dim,
            nhid=args.hidden_size,
            nclass=num_cats,
            nlayer=args.n_layers,
            dropout=args.dropout,
        )
    elif args.model == "gat":
        model = GATNodeCLF(
            in_channels=feat_dim, out_channels=num_cats, dropout=args.dropout,
        )
    else:
        # MORE MODELS
        raise Exception("such model does not exist !")
    trainer = FedNodeClfTrainer(model)
    logging.info("Model and Trainer  - done")
    return model, trainer


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, num_cats, feat_dim = load_data(args)

    # load model
    model, trainer = create_model(args, feat_dim, num_cats)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer)
    fedml_runner.run()
