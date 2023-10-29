from .fed_camelyon16 import load_partition_fed_camelyon16


def load_data(args):
    dataset_name = args.dataset.lower()
    if dataset_name in ["fed_camelyon16", "fed-camelyon16"]:
        dataset = load_partition_fed_camelyon16(args)
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")

    return dataset
