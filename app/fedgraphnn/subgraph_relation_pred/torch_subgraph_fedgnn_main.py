import fedml
from data.data_loader import *
from model.rgcn import RGCNEncoder, DistMultDecoder
from trainer.fed_subgraph_rel_trainer import FedSubgraphRelTrainer
from torch_geometric.nn import GAE
from fedml.simulation import SimulatorMPI

def load_data(args):
    if args.dataset not in ["YAGO3-10", "wn18rr", "FB15k-237"]:
        raise Exception("no such dataset!")

    args.part_algo = "Louvain"
    args.pred_task = "relation"
    compact = args.model == "graphsage"

    args.metric = "AP"

    unif = True if args.partition_method == "homo" else False

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
        None
    ]

    return dataset


def create_model(model_name):
    logging.info("create_model. model_name = %s" % (model_name))
    if model_name == "rgcn":
        model = GAE(
            RGCNEncoder(34662, hidden_channels=500, num_relations=11),
            DistMultDecoder(11 // 2, hidden_channels=500),
        )
    else:
        raise Exception("such model does not exist !")
    trainer = FedSubgraphRelTrainer(model)
    logging.info("Model and Trainer  - done")
    return model, trainer



if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset = load_data(args)

    # load model
    model, trainer = create_model(args.model)

    # start training
    simulator = SimulatorMPI(args, device, dataset, model, trainer)
    simulator.run()
