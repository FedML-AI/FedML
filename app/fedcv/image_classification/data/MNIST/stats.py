'''
assumes that the user has already generated .json file(s) containing data
'''

import argparse
import json
import os

import numpy as np
from scipy import stats

parser = argparse.ArgumentParser()

parser.add_argument('--name',
                    help='name of dataset to parse; default: sent140;',
                    type=str,
                    default='sent140')

parser.add_argument('--subdir',
                    help='train or test',
                    type=str,
                    default='train')

args = parser.parse_args()


def load_data(name):
    users = []
    num_samples = []

    parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(parent_path, name)
    subdir = os.path.join(data_dir, args.subdir)

    files = os.listdir(subdir)
    files = [f for f in files if f.endswith('.json')]

    for f in files:
        file_dir = os.path.join(subdir, f)

        with open(file_dir) as inf:
            data = json.load(inf)

        users.extend(data['users'])
        num_samples.extend(data['num_samples'])

    return users, num_samples


def print_dataset_stats(name):
    users, num_samples = load_data(name)
    num_users = len(users)

    print('####################################')
    print('DATASET: %s' % name)
    print('%d users' % num_users)
    print('%d samples (total)' % np.sum(num_samples))
    print('%.2f samples per user (mean)' % np.mean(num_samples))
    print('num_samples (std): %.2f' % np.std(num_samples))
    print('num_samples (std/mean): %.2f' % (np.std(num_samples) / np.mean(num_samples)))
    print('num_samples (skewness): %.2f' % stats.skew(num_samples))

    bins = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    if args.name == 'shakespeare':
        bins = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
    if args.name == 'nist':
        bins = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420,
                440, 460, 480, 500]

    hist, edges = np.histogram(num_samples, bins=bins)
    print("\nnum_sam\tnum_users")
    for e, h in zip(edges, hist):
        print(e, "\t", h)

    # parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # data_dir = os.path.join(parent_path, 'data', name, 'data')

    # plt.hist(num_samples, bins=bins)
    # fig_name = "%s_hist_nolabel.png" % name
    # fig_dir = os.path.join(data_dir, fig_name)
    # plt.savefig(fig_dir)
    # plt.title(name)
    # plt.xlabel("number of samples")
    # plt.ylabel("number of users")
    # fig_name = "%s_hist.png" % name
    # fig_dir = os.path.join(data_dir, fig_name)
    # plt.savefig(fig_dir)


print_dataset_stats(args.name)
