import fedml
from fedml.simulation import SimulatorMPI
from fedml.data.data_loader import load_poisoned_dataset_from_edge_case_examples

from fedml.data.edge_case_examples.data_loader import download_edgecase_data


def load_poisoned_data(args):
    fedml.logging.info("load_data. dataset_name = %s" % args.dataset)
    download_edgecase_data(args.data_cache_dir)

    (
        poisoned_train_loader,
        targetted_task_test_loader,
        num_dps_poisoned_dataset,
    ) = load_poisoned_dataset_from_edge_case_examples(args)

    dataset, class_num = fedml.data.load(args)

    [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ] = dataset

    dataset_rt = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        poisoned_train_loader,
        targetted_task_test_loader,
        num_dps_poisoned_dataset,
    ]
    class_num = dataset[7]
    return dataset_rt, class_num


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_poisoned_data(args)

    model = fedml.model.create(args, output_dim)

    # start training
    simulator = SimulatorMPI(args, device, dataset, model)
    simulator.run()
