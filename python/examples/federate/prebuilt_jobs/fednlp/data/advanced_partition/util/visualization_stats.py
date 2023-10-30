import statistics
import os
import h5py
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "--partition_name", type=str, metavar="PN", help="name of the method "
)
parser.add_argument(
    "--partition_file",
    type=str,
    default="data/partition_files/wikiner_partition.h5",
    metavar="DF",
    help="data file path",
)
parser.add_argument("--task_name", type=str, metavar="TN", help="task name")

parser.add_argument(
    "--figure_path", type=str, metavar="TN", help="the place to store generated figures"
)


args = parser.parse_args()

f = h5py.File(args.partition_file, "r")

partition_samples = []
partition_total = 0
partition_name = ""
client_number = 0

for i in f.keys():
    if args.partition_name in i:
        partition_name = i
        client_number = f[i + "/n_clients"][()]
        break

partition_data_path = "/" + partition_name + "/partition_data/"

for i in f[partition_data_path].keys():
    train_path = partition_data_path + str(i) + "/train/"
    test_path = partition_data_path + str(i) + "/test/"
    partition_samples.append(len(f[train_path][()]) + len(f[test_path][()]))
    partition_total = partition_total + len(f[train_path][()]) + len(f[test_path][()])


f.close()


print("")
print("users")
print(client_number)

print("sample total")
print(partition_total)

print("sample mean")
mean = partition_total / client_number
print(mean)

print("std")
std = statistics.stdev(partition_samples)
print(std)

print("std/mean")
print(std / mean)

data_dir = args.figure_path
plt.hist(partition_samples)
plt.title(args.task_name + " " + args.partition_name)
plt.xlabel("number of samples")
plt.ylabel("number of clients")
fig_name = args.task_name + "_%s_hist.png" % args.partition_name
fig_dir = os.path.join(data_dir, fig_name)
plt.savefig(fig_dir)
