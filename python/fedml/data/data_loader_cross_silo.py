from torch.utils import data
from .data_loader import load_synthetic_data


def load_cross_silo(args):
    return load_synthetic_data_cross_silo(args)


def split_array(array, n_dist_trainer):
    r = len(array) % n_dist_trainer
    if r != 0:
        for _ in range(n_dist_trainer - r):
            array.append(array[-1])
    split_array = []
    chuhck_size = len(array) // n_dist_trainer

    for i in range(n_dist_trainer):
        split_array.append(array[i * chuhck_size : (i + 1) * chuhck_size])
    return split_array


def split_dl(dl, n_dist_trainer):
    ds = dl.dataset
    bs = dl.batch_size
    split_dl = []
    if isinstance(dl.sampler, data.RandomSampler):
        shuffle = True
    else:
        shuffle = False
    for rank in range(n_dist_trainer):
        sampler = data.distributed.DistributedSampler(
            dataset=ds,
            num_replicas=n_dist_trainer,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
        process_dl = data.DataLoader(
            dataset=dl.dataset,
            batch_size=bs,
            shuffle=False,
            drop_last=False,
            sampler=sampler,
        )
        split_dl.append(process_dl)
    return split_dl


def split_data_for_dist_trainers(data_loaders, n_dist_trainer):
    for index, dl in data_loaders.items():
        if isinstance(dl, data.DataLoader):
            data_loaders[index] = split_dl(dl, n_dist_trainer)
        else:
            data_loaders[index] = split_array(dl, n_dist_trainer)
    return data_loaders


def load_synthetic_data_cross_silo(args):
    n_dist_trainer = args.n_proc_in_silo
    dataset, class_num = load_synthetic_data(args)
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

    train_data_local_dict = split_data_for_dist_trainers(
        train_data_local_dict, n_dist_trainer
    )

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num
