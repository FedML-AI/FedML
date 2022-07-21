from torch.utils import data

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


