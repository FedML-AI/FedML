import os
import h5py
import argparse
import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance

parser = argparse.ArgumentParser()

parser.add_argument(
    "--partition_file",
    type=str,
    default="data/partition_files/wikiner_partition.h5",
    metavar="PF",
    help="data partition path",
)
parser.add_argument(
    "--data_file",
    type=str,
    default="data/data_files/wikiner_data.h5",
    metavar="DF",
    help="data file path",
)
parser.add_argument("--task_name", type=str, metavar="TN", help="task name")

parser.add_argument(
    "--cluster_num", type=int, metavar="KN", help="cluster of partition"
)

parser.add_argument(
    "--client_number",
    type=int,
    metavar="CN",
    help="client number of this partition method",
)

parser.add_argument(
    "--figure_path", type=str, metavar="TN", help="the place to store generated figures"
)

parser.add_argument(
    "--task_type",
    type=str,
    default="name entity recognition",
    metavar="TT",
    help="task type",
)

args = parser.parse_args()


f = h5py.File(args.partition_file, "r")

quantity_partition_methods = []


for i in f.keys():
    if "niid_quantity" in i:
        quantity_partition_methods.append(i)

f.close()

quantity_partition_methods = [
    "niid_quantity_beta=100.0",
    "niid_quantity_beta=10.0",
    "niid_quantity_beta=5.0",
    "niid_quantity_beta=1.0",
]


a = np.repeat("100", args.client_number)
b = np.repeat("10", args.client_number)
c = np.repeat("5", args.client_number)
d = np.repeat("1", args.client_number)


alphas = np.concatenate((a, b, c, d), axis=None)


temp = "kmeans_" + str(args.cluster_num)
client_assignment = np.array([])


client_quantity = []
partition_file = h5py.File(args.partition_file, "r")

for index, partition in enumerate(quantity_partition_methods):

    partition_data_path = "/" + partition + "/partition_data/"
    for client_id in partition_file[partition_data_path].keys():
        client_len = len(
            partition_file[partition_data_path][client_id]["train"][()]
        ) + len(partition_file[partition_data_path][client_id]["test"][()])
        client_quantity.append(client_len)


partition_file.close()


df = pd.DataFrame({"client_length": client_quantity, "beta": alphas})


data_dir = args.figure_path
plt.figure(figsize=(20, 15))
g = sns.displot(df, x="client_length", hue="beta", kind="kde", common_norm=False)
g.set(xlabel="", ylabel="")
g.tight_layout(pad=0.1)
plt.xlim(0, 800)
fig_name = args.task_name + "_all_partition_quantity_distplot.pdf"
fig_dir = os.path.join(data_dir, fig_name)

plt.savefig(fig_dir, bbox_inches="tight", format="pdf")
