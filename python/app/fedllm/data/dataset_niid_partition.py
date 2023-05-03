from typing import (
    List,
    MutableSequence,
    Optional,
    Sequence,
)

from argparse import ArgumentParser, Namespace
from pathlib import Path

from datasets import Dataset, load_dataset
import numpy as np
from tqdm import tqdm
from transformers import set_seed


def _get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--client_number",
        type=int,
        default=100,
        metavar="CN",
        help="client number for lda partition"
    )
    parser.add_argument(
        "--client_start_idx",
        type=int,
        default=1,
        help="starting client index"
    )
    parser.add_argument(
        "-i",
        "--dataset_path",
        type=str,
        required=True,
        metavar="DF",
        help="dataset path",
        nargs="+"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        metavar="O",
        help="dataset output path",
        nargs="+"
    )
    parser.add_argument(
        "--test_dataset_size",
        type=int,
        default=100,
        metavar="TDS",
        help="test dataset size"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        metavar="A",
        help="alpha value for quantity skew"
    )
    parser.add_argument(
        "--seed",
        # Optional[int]
        type=lambda s: None if s.lower().strip() in {"", "none", "null"} else int(s),
        default=None,
        help="random seed"
    )

    return parser


def partition_class_samples_with_dirichlet_distribution(
        dataset_length: int,
        alpha: float,
        client_num: int,
        idx_batch: Sequence[Sequence[int]],
        idx_k: MutableSequence[int]
) -> List[List[int]]:
    """

    Args:
        dataset_length: total length of the dataset
        alpha: coefficient controlling the similarity of each client, the larger
            the alpha the similar data for each client
        client_num: number of clients
        idx_batch: 2D list of shape(client_num, ?), this is the list of index for each client
        idx_k: 1D list of index of the dataset

    Returns:
        idx_batch: 2D list shape(client_num, ?) list of index for each client

    """

    # first shuffle the index
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array(
        [p * (len(idx_j) < dataset_length / client_num) for p, idx_j in zip(proportions, idx_batch)]
    )
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the new batch list for each client
    idx_batch = [
        list(idx_j) + idx.tolist()
        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
    ]

    return idx_batch


def save_partition(
        dataset: Dataset,
        partition_idx: List[List[int]],
        output_base_path: str,
        seed: Optional[int],
        client_start_idx: int = 0
) -> None:
    output_base_path = Path(output_base_path)

    for client_idx, indices in enumerate(partition_idx, start=client_start_idx):
        filename = f"{output_base_path.stem}-client_idx={client_idx}" \
                   f",max_client={len(partition_idx)},seed={seed}" \
                   f"{output_base_path.suffix}"

        output_path = output_base_path.parent / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.select(indices).to_json(output_path)


def main(args: Namespace) -> None:
    assert args.client_number > 0
    assert args.test_dataset_size >= 0
    assert len(args.dataset_path) == len(args.output_path)

    if args.seed is not None:
        print(f"set seed to {args.seed}")
        set_seed(args.seed)

    if len(args.dataset_path) == 1:
        train_output_path = test_output_path = Path(args.output_path[0])
        train_output_path = train_output_path.parent / f"train_{train_output_path.name}"
        test_output_path = test_output_path.parent / f"test_{test_output_path.name}"

        dataset = load_dataset("json", data_files=args.dataset_path[0])["train"]
        dataset = dataset.train_test_split(test_size=args.test_dataset_size, seed=args.seed)
    else:
        train_output_path, test_output_path, *_ = args.output_path

        dataset = load_dataset("json", data_files={"train": args.dataset_path[0], "test": args.dataset_path[1]})
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_index_list = list(range(len(train_dataset)))
    test_index_list = list(range(len(test_dataset)))

    min_train_size = 0
    min_test_size = 0

    train_partition_idx = []
    test_partition_idx = []

    print("start dirichlet distribution")
    with tqdm(desc="Partitioning...") as p_bar:
        while min_test_size < 1 or min_train_size < 1:
            train_partition_idx = [[] for _ in range(args.client_number)]
            test_partition_idx = [[] for _ in range(args.client_number)]
            train_n = len(train_index_list)
            test_n = len(test_index_list)
            train_partition_idx = partition_class_samples_with_dirichlet_distribution(
                train_n, args.alpha, args.client_number, train_partition_idx, train_index_list
            )
            test_partition_idx = partition_class_samples_with_dirichlet_distribution(
                test_n, args.alpha, args.client_number, test_partition_idx, test_index_list
            )

            min_train_size = min(len(i) for i in train_partition_idx)
            min_test_size = min(len(i) for i in test_partition_idx)

            # update progress bar
            p_bar.update(1)

    assert len(train_partition_idx) == len(test_partition_idx) == args.client_number
    print(f"minsize of the train data: {min_train_size:,}")
    print(f"minsize of the test data {min_test_size:,}")

    save_partition(train_dataset, train_partition_idx, train_output_path, args.seed, args.client_start_idx)
    save_partition(test_dataset, test_partition_idx, test_output_path, args.seed, args.client_start_idx)


if __name__ == '__main__':
    main(args=_get_arg_parser().parse_args())
