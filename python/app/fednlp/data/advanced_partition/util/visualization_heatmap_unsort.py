import os
import h5py
import argparse
import pandas as pd
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance

parser = argparse.ArgumentParser()

parser.add_argument(
    "--partition_name", type=str, metavar="PN", help="name of the method "
)
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

temp = "kmeans_" + str(args.cluster_num)
client_assignment = []

if args.task_type == "text_classification":
    data = h5py.File(args.data_file, "r")
    total_labels = [data["Y"][i][()] for i in data["Y"].keys()]
    attributes = json.loads(data["attributes"][()])
    label_vocab = attributes["label_vocab"]
    client_assignment = [label_vocab[label] for label in total_labels]
    data.close()
else:
    f = h5py.File(args.partition_file, "r")
    for i in f.keys():
        if temp in i:
            client_assignment = f[i + "/client_assignment/"][()]
            break
    f.close()
partition_data_path = "/" + args.partition_name + "/partition_data/"

client_numbers = args.client_number
client_index = list(range(client_numbers))
print(client_index)
client_data_distribution = []
cluster_num = len(set(client_assignment))


f = h5py.File(args.partition_file, "r")

for i in client_index:
    temp = []
    single_client_data = []
    probability_array = np.zeros(cluster_num)
    temp.extend(f[partition_data_path + str(i) + "/train"][()])
    temp.extend(f[partition_data_path + str(i) + "/test"][()])
    single_client_data = np.array([client_assignment[i] for i in temp])
    unique, counts = np.unique(single_client_data, return_counts=True)
    for key, value in dict(zip(unique, counts)).items():
        probability_array[key] = value
    client_data_distribution.append(probability_array)
f.close()
heat_map_data = np.zeros((client_numbers, client_numbers))

for i in range(client_numbers):
    for j in range(client_numbers):
        heat_map_data[i][j] = distance.jensenshannon(
            client_data_distribution[i], client_data_distribution[j]
        )
""" #reorder index based on the sum of distance in each client
client_data_distribution_reorder_index = [np.where(np.all(heat_map_data == i,axis = 1))[0][0] for i in sorted(heat_map_data, key=lambda client: sum(client), reverse=True)]
#reorder the matrix based on the reorder index
for index, value in enumerate(heat_map_data):
    heat_map_data[index] = value[client_data_distribution_reorder_index]
heat_map_data = heat_map_data[client_data_distribution_reorder_index] """


client_sum_order = sorted([sum(i) for i in heat_map_data], reverse=True)

data_dir = args.figure_path
fig_name = args.task_name + "_%s_clients_heatmap_unsort.png" % args.partition_name
fig_dir = os.path.join(data_dir, fig_name)
fig_dims = (30, 22)
fig, ax = plt.subplots(figsize=fig_dims)
sns.set(font_scale=6)
sns.heatmap(heat_map_data, linewidths=0.05, cmap="Blues", cbar=True, vmin=0, vmax=0.8)
ax.tick_params(
    labelbottom=False,
    labelleft=False,
    labeltop=False,
    left=False,
    bottom=False,
    top=False,
)
fig.tight_layout(pad=0.1)
plt.savefig(fig_dir)

sns.set(font_scale=1)

plt.figure(figsize=(20, 15))
fig = sns.distplot(client_sum_order)
plt.xlim(0, None)
plt.xlabel("distance")
plt.xticks(fig.get_xticks(), fig.get_xticks() / 100)
fig_name = args.task_name + "_%s_clients_sum_distplot_unsort.png" % args.partition_name
fig_dir = os.path.join(data_dir, fig_name)
plt.title(args.task_name + "_%s_clients_sum_distplot_unsort" % args.partition_name)
plt.savefig(fig_dir, bbox_inches="tight")
