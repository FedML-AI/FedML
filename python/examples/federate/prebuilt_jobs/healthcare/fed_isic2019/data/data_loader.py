from .fed_isic2019 import load_partition_fed_isic2019


def load_data(args):
    dataset_name = args.dataset.lower()
    if dataset_name in ["fed_isic2019", "fed-isic2019"]:
        dataset = load_partition_fed_isic2019(args)
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")

    return dataset
