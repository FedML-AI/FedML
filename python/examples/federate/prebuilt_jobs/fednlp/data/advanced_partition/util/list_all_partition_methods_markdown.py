import h5py
import json
import os
from pytablewriter import MarkdownTableWriter

all_partition_files = [
    "20news_partition.h5",
    "agnews_partition.h5",
    "cornell_movie_dialogue_partition.h5",
    "onto_partition.h5",
    "ploner_partition.h5",
    "squad_1.1_partition.h5",
    "sst_2_partition.h5",
    "wikiner_partition.h5",
    "w_nut_partition.h5",
]


def main():
    partition_methods = []
    for partition_file_name in all_partition_files:
        each_partition = []
        file_name = "data/partition_files/" + partition_file_name
        partition = h5py.File(file_name, "r")
        each_partition.append(partition_file_name[:-13])
        each_partition.append(list(partition.keys()))
        partition_methods.append(each_partition)
        partition.close()
    writer = MarkdownTableWriter(
        table_name="partition_methods_table",
        headers=[
            "dataset_name",
            "partition_methods existed in provided partition files",
        ],
        value_matrix=partition_methods,
    )
    writer.write_table()


if __name__ == "__main__":
    main()

exit()
