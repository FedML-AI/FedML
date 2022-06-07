import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from data_preprocessing.cityscapes.data_loader import partition_data as partition_cityscapes
from data_preprocessing.pascal_voc_augmented.data_loader import partition_data as partition_pascal

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5, help='Partition Alpha')
    parser.add_argument('--data_dir', type=str, default='/home/chaoyanghe/BruteForce/FedML/data/pascal_voc'
                                                        '/benchmark_RELEASE', help='Dataset directory')
    parser.add_argument('--dataset', type=str, default='pascal_voc', help="Name of dataset")

    args = parser.parse_args()

    # alpha = 100
    # data_dir = "/content/data/benchmark_RELEASE/"
    if args.dataset == 'pascal_voc':
        net_data_idx_map, train_data_cls_counts = partition_pascal(args.data_dir, "hetero", 4, args.alpha, 513)
    elif args.dataset == 'cityscapes':
        net_data_idx_map, train_data_cls_counts = partition_cityscapes(args.data_dir, "hetero", 4, args.alpha, 513)
    else:
        raise NotImplementedError

    clients = train_data_cls_counts[0].keys()
    client1 = np.array(list(train_data_cls_counts[0].values()))
    client2 = np.array(list(train_data_cls_counts[1].values()))
    client3 = np.array(list(train_data_cls_counts[2].values()))
    client4 = np.array(list(train_data_cls_counts[3].values()))
    # Add more clients if necessary

    total = client1 + client2 + client3 + client4
    proportion_client1 = np.true_divide(client1, total) * 100
    proportion_client2 = np.true_divide(client2, total) * 100
    proportion_client3 = np.true_divide(client3, total) * 100
    proportion_client4 = np.true_divide(client4, total) * 100
    ind = [x for x, _ in enumerate(clients)]

    plt.bar(ind, proportion_client4, width=0.8, label='c4', color='b',
            bottom=proportion_client1 + proportion_client2 + proportion_client3)
    plt.bar(ind, proportion_client3, width=0.8, label='c3', color='g',
            bottom=proportion_client1 + proportion_client2)
    plt.bar(ind, proportion_client2, width=0.8, label='c2', color='silver', bottom=proportion_client1)
    plt.bar(ind, proportion_client1, width=0.8, label='c1', color='gold')

    plt.xticks(ind, clients)
    plt.ylabel("Data")
    plt.xlabel("Classes")
    plt.title("Class Distribution: clients=4, alpha={}".format(args.alpha))
    plt.ylim = 1.0

    # rotate axis labels
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.show()
    plt.savefig(args.dataset + '.png')

