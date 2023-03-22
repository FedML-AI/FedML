from .fed_tcga_brca import load_partition_fed_tcga_brca


def load_data(args):
    dataset_name = args.dataset.lower()
    if dataset_name in ["fed_tcga_brca", "fed-tcga-brca"]:
        dataset = load_partition_fed_tcga_brca(args)
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented")

    return dataset
