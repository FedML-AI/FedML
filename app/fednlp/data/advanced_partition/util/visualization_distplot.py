import os
import h5py
import argparse
import pandas as pd
import numpy as np
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

tc_partition_methods = []
non_tc_partition_methods = []


for i in f.keys():
    if "niid_label" in i:
        tc_partition_methods.append(i)
    if "niid_cluster" in i:
        non_tc_partition_methods.append(i)

f.close()

a = np.repeat("100", args.client_number)
b = np.repeat("10", args.client_number)
c = np.repeat("5", args.client_number)
d = np.repeat("1", args.client_number)

alphas = np.concatenate((a, b, c, d), axis=None)


temp = "kmeans_" + str(args.cluster_num)
client_assignment = np.array([])

if args.task_type == "text_classification":
    data = h5py.File(args.data_file, "r")
    client_assignment = [data["Y"][i][()] for i in data["Y"].keys()]
    for index, value in enumerate(set(client_assignment)):
        client_assignment = [index if i == value else i for i in client_assignment]
    data.close()
else:
    f = h5py.File(args.partition_file, "r")
    for i in f.keys():
        if temp in i:
            client_assignment = f[i + "/client_assignment/"][()]
            break
    f.close()

client_sum_order = np.array([])
for index, partition in enumerate(tc_partition_methods):

    partition_data_path = "/" + partition + "/partition_data/"

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
            # client_sum_order = np.append(client_sum_order,[heat_map_data[i][j]],axis=0)
    client_sum_order = np.append(
        client_sum_order, np.array([sum(i) for i in heat_map_data]), axis=0
    )


plot_array = []


df = pd.DataFrame({"label_sum": client_sum_order, "alpha": alphas})


data_dir = args.figure_path
plt.figure(figsize=(20, 15))
g = sns.displot(
    df, x="label_sum", hue="alpha", kind="kde", common_norm=False, fill=True
)


g.tight_layout(pad=0.1)
g.set(xlabel="", ylabel="")

plt.xlim(0, None)
fig_name = args.task_name + "_all_partition_label_distplot.pdf"
fig_dir = os.path.join(data_dir, fig_name)

plt.savefig(fig_dir, bbox_inches="tight", format="pdf")
