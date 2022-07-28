from .fed_lidc_idri import load_partition_fed_lidc_idri


def load_data(args):
    dataset_name = args.dataset.lower()
    if dataset_name in ["fed_lidc_idri", "fed-lidc-idri"]:
        dataset = load_partition_fed_lidc_idri(args)
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")

    return dataset
