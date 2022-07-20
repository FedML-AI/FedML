import logging

import fedml
from data.data_loader import load_partition_data, get_data
from fedml import FedMLRunner
from model.gcn_link import GCNLinkPred
from trainer.federated_lp_trainer import FedLinkPredTrainer


def load_data(args, dataset_name):
    if args.dataset not in [
        "CS",
        "Physics",
        "cora",
        "Cora_ML",
        "CiteSeer",
        "DBLP",
        "PubMed",
    ]:
        raise Exception("no such dataset!")
    elif args.dataset in ["CS", "Physics"]:
        args.type_network = "coauthor"
    else:
        args.type_network = "citation"

    compact = args.model == "graphsage"

    args.metric = "AP"

    unif = True if args.partition_method == "homo" else False

    sgs, num_graphs, num_feats, num_cats = get_data(args.data_cache_dir, args.dataset)

    feat_dim = sgs[0].x.shape[1]
    del sgs

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


def create_model(model_name, feat_dim, num_cats):
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (model_name, num_cats)
    )
    if model_name == "gcn":
        model = GCNLinkPred(in_channels=feat_dim, out_channels=feat_dim)
    else:
        raise Exception("such model does not exist !")
    trainer = FedLinkPredTrainer(model)
    logging.info("Model and Trainer  - done")
    return model, trainer


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, num_cats, feat_dim = load_data(args, args.dataset)

    # load model
    model, trainer = create_model(args.model, feat_dim, num_cats)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer)
    fedml_runner.run()
