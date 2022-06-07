import logging

import fedml
from trainer.gat_readout_trainer_regression import GatMoleculeNetTrainer
from trainer.sage_readout_trainer_regression import SageMoleculeNetTrainer
from trainer.gcn_trainer_readout_regression import GcnMoleculeNetTrainer
from data.data_loader import load_partition_data, get_data
from fedml.simulation import SimulatorMPI
from model.gat_readout import GatMoleculeNet
from model.gcn_readout import GcnMoleculeNet
from model.sage_readout import SageMoleculeNet


def load_data(args, dataset_name):
    num_cats, feat_dim = 0, 0
    if dataset_name not in ["esol", "freesolv", "herg", "lipo", "qm9"]:
        raise Exception("no such dataset!")

    compact = args.model == "graphsage"

    logging.info("load_data. dataset_name = %s" % dataset_name)
    _, feature_matrices, labels = get_data(args.data_cache_dir + args.dataset)
    unif = True if args.partition_method == "homo" else False
    if args.model == "gcn":
        args.normalize_features = True
        args.normalize_adjacency = True

    args.metric = "rmse"

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
        args.data_cache_dir + args.dataset,
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
        labels[0].shape[0],
    ]

    return dataset, feature_matrices[0].shape[1], labels[0].shape[0]


def create_model(args, model_name, feat_dim, num_cats, output_dim):
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (model_name, output_dim)
    )
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
        trainer = SageMoleculeNetTrainer(model)
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
        trainer = GatMoleculeNetTrainer(model)
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
        trainer = GcnMoleculeNetTrainer(model)
    else:
        raise Exception("such model does not exist !")

    return model, trainer


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, feat_dim, num_cats = load_data(args, args.dataset)

    # load model
    model, trainer = create_model(args, args.model, feat_dim, num_cats, output_dim=None)

    # start training
    try:
        simulator = SimulatorMPI(args, device, dataset, model, trainer)
        simulator.run()
    except Exception as e:
        raise e
