import logging
import traceback

from mpi4py import MPI

import fedml
from data.data_loader import load_partition_data, get_data
from fedml.simulation import SimulatorMPI
from model.gin import GIN
from trainer.gin_trainer import GINSocialNetworkTrainer


def load_data(args, dataset_name):
    if args.dataset not in [
        "COLLAB",
        "REDDIT-BINARY",
        "REDDIT-MULTI-5K",
        "REDDIT-MULTI-12K",
        "IMDB-BINARY",
        "IMDB-MULTI",
    ]:
        raise Exception("no such dataset!")

    compact = args.model == "graphsage"

    logging.info("load_data. dataset_name = %s" % dataset_name)
    _, feat_dim, num_cats = get_data(args.data_cache_dir, args.dataset)
    unif = True if args.partition_method == "homo" else False

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
        num_cats,
    ]

    return dataset, feat_dim, num_cats


def create_model(args, model_name, feat_dim, num_cats, output_dim):
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (model_name, output_dim)
    )
    if model_name == "gin":
        model = GIN(
            nfeat=feat_dim,
            nhid=args.hidden_size,
            nclass=num_cats,
            nlayer=args.n_layers,
            dropout=args.dropout,
            eps=args.eps,
        )
        trainer = GINSocialNetworkTrainer(model)
    else:
        raise Exception("such model does not exist !")
    logging.info("done")
    return model, trainer


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, feat_dim, num_cats = load_data(args, args.dataset)

    # create model
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model, trainer = create_model(args, args.model, feat_dim, num_cats, output_dim=None)

    # start training
    try:
        simulator = SimulatorMPI(args, device, dataset, model, trainer)
        simulator.run()
    except Exception as e:
        logging.info('traceback.format_exc():\n%s' % traceback.format_exc())
        MPI.COMM_WORLD.Abort()
