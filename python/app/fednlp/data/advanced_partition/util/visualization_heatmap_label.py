import os
import h5py
import argparse
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

heat_map_data = []
if args.task_type == "text_classification":
    data = h5py.File(args.data_file, "r")
    total_labels = [data["Y"][i][()] for i in data["Y"].keys()]
    attributes = json.loads(data["attributes"][()])
    label_vocab = attributes["label_vocab"]
    client_assignment = [label_vocab[label] for label in total_labels]
    label_vocab_length = len(attributes["label_vocab"])
    heat_map_data = np.zeros((label_vocab_length, args.client_number))
    data.close()
else:
    f = h5py.File(args.partition_file, "r")
    for i in f.keys():
        if temp in i:
            client_assignment = f[i + "/client_assignment/"][()]
            break
    heat_map_data = np.zeros((args.cluster_num, args.client_number))

    f.close()
partition_data_path = "/" + args.partition_name + "/partition_data/"

f = h5py.File(args.partition_file, "r")


for client_id in f[partition_data_path].keys():

    data_in_single_client = []
    single_client_length = 0
    label_number_dict = dict()
    data_in_single_client.extend(f[partition_data_path][client_id]["train"][()])
    data_in_single_client.extend(f[partition_data_path][client_id]["test"][()])
    single_client_length = len(data_in_single_client)
    for index in data_in_single_client:
        label = client_assignment[index]
        if label not in label_number_dict:
            label_number_dict[label] = 1
        else:
            label_number_dict[label] += 1
    sort_labels = sorted([k for k in label_number_dict.keys()])

    for label_id in sort_labels:
        heat_map_data[label_id][[int(client_id)]] = (
            label_number_dict[label_id] / single_client_length
        )
f.close()

data_dir = args.figure_path
fig_name = args.task_name + "_%s_clients_heatmap_label.png" % args.partition_name
fig_dir = os.path.join(data_dir, fig_name)
fig_dims = (30, 10)
fig, ax = plt.subplots(figsize=fig_dims)
sns.set(font_scale=4)
sns.heatmap(heat_map_data, linewidths=0.05, cmap="YlGnBu", cbar=False, vmin=0, vmax=1)
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
