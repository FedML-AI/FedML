from .fed_ixi import load_partition_fed_ixi


def load_data(args):
    dataset_name = args.dataset.lower()
    if dataset_name in ["fed_ixi", "fed-ixi"]:
        dataset = load_partition_fed_ixi(args)
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")

    return dataset
