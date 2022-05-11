import logging

import numpy as np


def non_iid_partition_with_dirichlet_distribution(
    label_list, client_num, classes, alpha, task="classification"
):
    """
    Obtain sample index list for each client from the Dirichlet distribution.

    This LDA method is first proposed by :
    Measuring the Effects of Non-Identical Data Distribution for
    Federated Visual Classification (https://arxiv.org/pdf/1909.06335.pdf).

    This can generate nonIIDness with unbalance sample number in each label.
    The Dirichlet distribution is a density over a K dimensional vector p whose K components are positive and sum to 1.
    Dirichlet can support the probabilities of a K-way categorical event.
    In FL, we can view K clients' sample number obeys the Dirichlet distribution.
    For more details of the Dirichlet distribution, please check https://en.wikipedia.org/wiki/Dirichlet_distribution

    Parameters
    ----------
        label_list : the label list from classification/segmentation dataset
        client_num : number of clients
        classes: the number of classification (e.g., 10 for CIFAR-10) OR a list of segmentation categories
        alpha: a concentration parameter controlling the identicalness among clients.
        task: CV specific task eg. classification, segmentation
    Returns
    -------
        samples : ndarray,
            The drawn samples, of shape ``(size, k)``.
    """
    net_dataidx_map = {}
    K = classes

    # For multiclass labels, the list is ragged and not a numpy array
    N = len(label_list) if task == "segmentation" else label_list.shape[0]

    # guarantee the minimum number of sample in each client
    min_size = 0
    while min_size < 10:
        idx_batch = [[] for _ in range(client_num)]

        if task == "segmentation":
            # Unlike classification tasks, here, one instance may have multiple categories/classes
            for c, cat in enumerate(classes):
                if c > 0:
                    idx_k = np.asarray(
                        [
                            np.any(label_list[i] == cat)
                            and not np.any(np.in1d(label_list[i], classes[:c]))
                            for i in range(len(label_list))
                        ]
                    )
                else:
                    idx_k = np.asarray(
                        [np.any(label_list[i] == cat) for i in range(len(label_list))]
                    )

                # Get the indices of images that have category = c
                idx_k = np.where(idx_k)[0]
                (
                    idx_batch,
                    min_size,
                ) = partition_class_samples_with_dirichlet_distribution(
                    N, alpha, client_num, idx_batch, idx_k
                )
        else:
            # for each classification in the dataset
            for k in range(K):
                # get a list of batch indexes which are belong to label k
                idx_k = np.where(label_list == k)[0]
                (
                    idx_batch,
                    min_size,
                ) = partition_class_samples_with_dirichlet_distribution(
                    N, alpha, client_num, idx_batch, idx_k
                )
    for i in range(client_num):
        np.random.shuffle(idx_batch[i])
        net_dataidx_map[i] = idx_batch[i]

    return net_dataidx_map


def partition_class_samples_with_dirichlet_distribution(
    N, alpha, client_num, idx_batch, idx_k
):
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array(
        [p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)]
    )
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the batch list for each client
    idx_batch = [
        idx_j + idx.tolist()
        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
    ]
    min_size = min([len(idx_j) for idx_j in idx_batch])

    return idx_batch, min_size


def record_data_stats(y_train, net_dataidx_map, task="classification"):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = (
            np.unique(np.concatenate(y_train[dataidx]), return_counts=True)
            if task == "segmentation"
            else np.unique(y_train[dataidx], return_counts=True)
        )
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug("Data statistics: %s" % str(net_cls_counts))
    return net_cls_counts
