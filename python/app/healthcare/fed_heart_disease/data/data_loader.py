from .fed_heart_disease import load_partition_fed_heart_disease


def load_data(args):
    dataset_name = args.dataset.lower()
    if dataset_name in ["fed_heart_disease", "fed-heart-disease"]:
        dataset = load_partition_fed_heart_disease(args)
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")

    return dataset
