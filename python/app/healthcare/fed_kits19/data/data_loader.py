from .fed_kits19 import load_partition_fed_kits19


def load_data(args):
    dataset_name = args.dataset.lower()
    if dataset_name in ["fed_kits19", "fed_kits2019", "fed-kits2019", "fed-kits19"]:
        dataset = load_partition_fed_kits19(args)
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")

    return dataset
