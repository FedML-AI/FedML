from typing import Counter
import h5py
import argparse
import numpy as np
import json
import math
from decimal import *
import random
import math
from collections import Counter
from numpy.lib.shape_base import split


def dynamic_batch_fill(
    label_index_tracker, label_index_matrix, remaining_length, current_label_id
):
    """
    params
    ------------------------------------------------------------------------
    label_index_tracker : 1d numpy array track how many data each label has used
    label_index_matrix : 2d array list of indexs of each label
    remaining_length : int remaining empty space in current partition client list
    current_label_id : int current round label id
    ------------------------------------------------------------------------

    return
    ---------------------------------------------------------
    label_index_offset: dict  dictionary key is label id
    and value is the offset associated with this key
    ----------------------------------------------------------
    """
    remaining_unfiled = remaining_length
    label_index_offset = {}
    label_remain_length_dict = {}
    total_label_remain_length = 0
    # calculate total number of all the remaing labels and each label's remaining length
    for label_id, label_list in enumerate(label_index_matrix):
        if label_id == current_label_id:
            label_remain_length_dict[label_id] = 0
            continue
        label_remaining_count = len(label_list) - label_index_tracker[label_id]
        if label_remaining_count > 0:
            total_label_remain_length = (
                total_label_remain_length + label_remaining_count
            )
        else:
            label_remaining_count = 0
        label_remain_length_dict[label_id] = label_remaining_count
    length_pointer = remaining_unfiled

    if total_label_remain_length > 0:
        label_sorted_by_length = {
            k: v
            for k, v in sorted(
                label_remain_length_dict.items(), key=lambda item: item[1]
            )
        }
    else:
        label_index_offset = label_remain_length_dict
        return label_index_offset
    # for each label calculate the offset move forward by distribution of remaining labels
    for label_id in label_sorted_by_length.keys():
        fill_count = math.ceil(
            label_remain_length_dict[label_id]
            / total_label_remain_length
            * remaining_length
        )
        fill_count = min(fill_count, label_remain_length_dict[label_id])
        offset_forward = fill_count
        # if left room not enough for all offset set it to 0
        if length_pointer - offset_forward <= 0 and length_pointer > 0:
            label_index_offset[label_id] = length_pointer
            length_pointer = 0
            break
        else:
            length_pointer -= offset_forward
            label_remain_length_dict[label_id] -= offset_forward
        label_index_offset[label_id] = offset_forward

    # still has some room unfilled
    if length_pointer > 0:
        for label_id in label_sorted_by_length.keys():
            # make sure no infinite loop happens
            fill_count = math.ceil(
                label_sorted_by_length[label_id]
                / total_label_remain_length
                * length_pointer
            )
            fill_count = min(fill_count, label_remain_length_dict[label_id])
            offset_forward = fill_count
            if length_pointer - offset_forward <= 0 and length_pointer > 0:
                label_index_offset[label_id] += length_pointer
                length_pointer = 0
                break
            else:
                length_pointer -= offset_forward
                label_remain_length_dict[label_id] -= offset_forward
            label_index_offset[label_id] += offset_forward

    return label_index_offset


def label_skew_process(label_vocab, label_assignment, client_num, alpha, data_length):
    """
    params
    -------------------------------------------------------------------
    label_vocab : dict label vocabulary of the dataset
    label_assignment : 1d list a list of label, the index of list is the index associated to label
    client_num : int number of clients
    alpha : float similarity of each client, the larger the alpha the similar data for each client
    -------------------------------------------------------------------
    return
    ------------------------------------------------------------------
    partition_result : 2d array list of partition index of each client
    ------------------------------------------------------------------
    """
    label_index_matrix = [[] for _ in label_vocab]
    label_proportion = []
    partition_result = [[] for _ in range(client_num)]
    client_length = 0
    print("client_num", client_num)
    # shuffle indexs and calculate each label proportion of the dataset
    for index, value in enumerate(label_vocab):
        label_location = np.where(label_assignment == value)[0]
        label_proportion.append(len(label_location) / data_length)
        np.random.shuffle(label_location)
        label_index_matrix[index].extend(label_location[:])
    print("proportion", label_proportion)
    # calculate size for each partition client
    label_index_tracker = np.zeros(len(label_vocab), dtype=int)
    total_index = data_length
    each_client_index_length = int(total_index / client_num)
    print("each index length", each_client_index_length)
    client_dir_dis = np.array([alpha * l for l in label_proportion])
    print("alpha", alpha)
    print("client dir dis", client_dir_dis)
    proportions = np.random.dirichlet(client_dir_dis)
    print("dir distribution", proportions[0])
    # add all the unused data to the client
    for client_id in range(len(partition_result)):
        each_client_partition_result = partition_result[client_id]
        proportions = np.random.dirichlet(client_dir_dis)
        print(client_id, proportions)
        print(type(proportions[0]))
        while True in np.isnan(proportions):
            proportions = np.random.dirichlet(client_dir_dis)
        client_length = min(each_client_index_length, total_index)
        if total_index < client_length * 2:
            client_length = total_index
        total_index -= client_length
        client_length_pointer = client_length
        # for each label calculate the offset length assigned to by Dir distribution and then extend assignment
        for label_id, _ in enumerate(label_vocab):
            offset = round(proportions[label_id] * client_length)
            if offset >= client_length_pointer:
                offset = client_length_pointer
                client_length_pointer = 0
            else:
                if label_id == (len(label_vocab) - 1):
                    offset = client_length_pointer
                client_length_pointer -= offset

            start = int(label_index_tracker[label_id])
            end = int(label_index_tracker[label_id] + offset)
            label_data_length = len(label_index_matrix[label_id])
            # if the the label is assigned to a offset length that is more than what its remaining length
            if end > label_data_length:
                each_client_partition_result.extend(
                    label_index_matrix[label_id][start:]
                )
                label_index_tracker[label_id] = label_data_length
                label_index_offset = dynamic_batch_fill(
                    label_index_tracker,
                    label_index_matrix,
                    end - label_data_length,
                    label_id,
                )
                for fill_label_id in label_index_offset.keys():
                    start = label_index_tracker[fill_label_id]
                    end = (
                        label_index_tracker[fill_label_id]
                        + label_index_offset[fill_label_id]
                    )
                    each_client_partition_result.extend(
                        label_index_matrix[fill_label_id][start:end]
                    )
                    label_index_tracker[fill_label_id] = (
                        label_index_tracker[fill_label_id]
                        + label_index_offset[fill_label_id]
                    )
            else:
                each_client_partition_result.extend(
                    label_index_matrix[label_id][start:end]
                )
                label_index_tracker[label_id] = label_index_tracker[label_id] + offset

        # if last client still has empty rooms, fill empty rooms with the rest of the unused data
        if client_id == len(partition_result) - 1:
            print("last id length", len(each_client_partition_result))
            print("Last client fill the rest of the unfilled lables.")
            for not_fillall_label_id in range(len(label_vocab)):
                if label_index_tracker[not_fillall_label_id] < len(
                    label_index_matrix[not_fillall_label_id]
                ):
                    print("fill more id", not_fillall_label_id)
                    start = label_index_tracker[not_fillall_label_id]
                    each_client_partition_result.extend(
                        label_index_matrix[not_fillall_label_id][start:]
                    )
                    label_index_tracker[not_fillall_label_id] = len(
                        label_index_matrix[not_fillall_label_id]
                    )
        partition_result[client_id] = each_client_partition_result

    return partition_result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--client_number",
        type=int,
        default="100",
        metavar="CN",
        help="client number for lda partition",
    )

    parser.add_argument(
        "--data_file",
        type=str,
        default="data/data_files/20news_data.h5",
        metavar="DF",
        help="data pickle file path",
    )

    parser.add_argument(
        "--partition_file",
        type=str,
        default="data/partition_files/20news_partition.h5",
        metavar="PF",
        help="partition pickle file path",
    )

    parser.add_argument(
        "--task_type",
        type=str,
        metavar="TT",
        help="task type: [text_classification,reading_comprehension,sequence_tagging,sequence_to_sequence]",
    )

    parser.add_argument(
        "--skew_type", type=str, metavar="TT", help="skeq type: [label, feature]"
    )

    parser.add_argument("--seed", type=int, metavar="RS", help="random seed")

    parser.add_argument(
        "--kmeans_num", type=int, metavar="KN", help="number of k-means cluster"
    )

    parser.add_argument("--alpha", type=float, metavar="A", help="alpha value for LDA")

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("start reading data")
    client_num = args.client_number
    alpha = args.alpha  # need adjustment for each dataset
    label_vocab_train = []
    label_vocab_test = []
    label_assignment = []
    train_length = 0
    test_length = 0
    partition_result_train = []
    partition_result_test = []

    print("retrieve data")
    # retrieve total index length
    data = h5py.File(args.data_file, "r")
    attributes = json.loads(data["attributes"][()])
    print("attributes", attributes.keys())
    total_index_list = attributes["index_list"]
    test_index_list = []
    train_index_list = []

    if "train_index_list" in attributes:
        test_index_list = attributes["test_index_list"]
        print(len(test_index_list))
        train_index_list = attributes["train_index_list"]
        print(len(train_index_list))
    else:
        # some dataset like wikiner do not have presplited train test dataset so we split the data
        train_length = int(len(total_index_list) * 0.9)
        train_index_list = total_index_list[0:train_length]
        test_index_list = total_index_list[train_length:]
    label_assignment_train = []
    label_assignment_test = []
    # retreive label vocab and label assigment the index of label assignment is the index of data assigned to this label
    # the value of each index is the label
    # label assignment's index all the index of the data and the label_assignment[index] stands for the label correspond to that index
    if args.skew_type == "label":
        if args.task_type == "text_classification":
            label_vocab_train = attributes["label_vocab"].keys()
            label_vocab_test = attributes["label_vocab"].keys()
            print(attributes["label_vocab"])
            label_assignment = np.array(
                [data["Y"][str(i)][()].decode("utf-8") for i in total_index_list]
            )
            label_assignment_test = np.array(
                [data["Y"][str(idx)][()].decode("utf-8") for idx in test_index_list]
            )
            label_assignment_train = np.array(
                [data["Y"][str(idx)][()].decode("utf-8") for idx in train_index_list]
            )
            train_length = len(label_assignment_train)
            test_length = len(label_assignment_test)
        elif args.task_type == "sequence_tagging":
            # TODO: convert seq of tags --> a str for the sorted set of tags
            # e.g.,  "OOOO B-PER I-PER  OOO B-LOC OOO B-LOC " ---> set{B-PER, B-LOC} --sorted--> "LOC-PER"
            # print(len(differnt types of pseudo-label))
            train_label_vocab_dict = dict()
            test_label_vocab_dict = dict()
            blacklist = [
                "CARDINAL",
                "DATE",
                "MONEY",
                "QUANTITY",
                "ORDINAL",
                "TIME",
                "LANGUAGE",
                "WORK_OF_ART",
                "LAW",
                "PERCENT",
            ]
            label = ""
            for index in total_index_list:
                tags = filter(
                    lambda x: x != "O",
                    [token.decode("utf-8") for token in data["Y"][str(index)][()]],
                )
                label_tags = set([t.split("-")[1] for t in tags])
                label_tags -= set(blacklist)
                label = "-".join(sorted(label_tags))
                if label == "":
                    label = "NULL"
                label_assignment.append(label)
            label_assignment = np.array(label_assignment)

            train_length = len(train_index_list)
            test_length = len(test_index_list)

            for index, value in enumerate(train_index_list):
                tags = filter(
                    lambda x: x != "O",
                    [token.decode("utf-8") for token in data["Y"][str(value)][()]],
                )
                label_tags = set([t.split("-")[1] for t in tags])
                label_tags -= set(blacklist)
                label = "-".join(sorted(label_tags))
                if label == "":
                    label = "NULL"
                if label in train_label_vocab_dict:
                    train_label_vocab_dict[label] += 1
                else:
                    train_label_vocab_dict[label] = 1
                label_assignment_train.append(label)
            label_assignment_train = np.array(label_assignment_train)

            for index, value in enumerate(test_index_list):
                tags = filter(
                    lambda x: x != "O",
                    [token.decode("utf-8") for token in data["Y"][str(value)][()]],
                )
                label_tags = set([t.split("-")[1] for t in tags])
                label_tags -= set(blacklist)
                label = "-".join(sorted(label_tags))
                if label == "":
                    label = "NULL"
                if label in test_label_vocab_dict:
                    test_label_vocab_dict[label] += 1
                else:
                    test_label_vocab_dict[label] = 1
                label_assignment_test.append(label)
            label_assignment_test = np.array(label_assignment_test)

            label_vocab_train = train_label_vocab_dict.keys()
            label_vocab_test = test_label_vocab_dict.keys()

            print("label vocab", len(set(label_assignment)))
            exit()

        elif args.task_type == "reading_comprehension":
            label_vocab = sorted(list(set(attributes["label_index_list"])))
            label_vocab_train = label_vocab
            label_vocab_test = label_vocab
            label_assignment = attributes["label_index_list"]
            label_assignment_test = np.array(
                [label_assignment[int(idx)] for idx in test_index_list]
            )
            label_assignment_train = np.array(
                [label_assignment[int(idx)] for idx in train_index_list]
            )
            train_length = len(label_assignment_train)
            test_length = len(label_assignment_test)
            # print("label assignment train set", Counter(label_assignment_train))
            # print("label assignment test set", Counter(label_assignment_test))

            label_assignment_train = np.array(label_assignment_train)
            label_assignment_test = np.array(label_assignment_test)

        else:
            print("Not Implemented.")
            exit()
    elif args.skew_type == "feature":
        # input feature skew --> Kmeans clustering + dir.
        partition = h5py.File(args.partition_file, "r")
        label_vocab_train = [i for i in range(args.kmeans_num)]
        label_vocab_test = [i for i in range(args.kmeans_num)]
        label_assignment = np.array(
            partition["kmeans_clusters=%d" % args.kmeans_num + "/client_assignment"][()]
        )
        label_assignment_test = np.array(
            [label_assignment[int(idx)] for idx in test_index_list]
        )
        label_assignment_train = np.array(
            [label_assignment[int(idx)] for idx in train_index_list]
        )
        train_length = len(label_assignment_train)
        test_length = len(label_assignment_test)
        partition.close()
    elif args.skew_type == "natural":
        partition = h5py.File(args.partition_file, "r")
        label_vocab = sorted(list(set(attributes["label_index_list"])))
        label_index_train_dict = dict()
        label_index_test_dict = dict()
        label_assignment = attributes["label_index_list"]
        for index, label in enumerate(label_assignment):
            if index < len(train_index_list):
                if label in label_index_train_dict:
                    label_index_train_dict[label].append(index)
                else:
                    label_index_train_dict[label] = [index]
            else:
                if label in label_index_test_dict:
                    label_index_test_dict[label].append(index)
                else:
                    label_index_test_dict[label] = [index]
        for label in label_vocab:
            partition_result_train.append(label_index_train_dict[label])
            partition_result_test.append(label_index_test_dict[label])
        partition.close()
    data.close()

    assert len(total_index_list) == len(label_assignment)
    if args.skew_type != "natural":
        print("start train data processing")

        partition_result_train = label_skew_process(
            label_vocab_train, label_assignment_train, client_num, alpha, train_length
        )
        print("start test data processing")
        partition_result_test = label_skew_process(
            label_vocab_test, label_assignment_test, client_num, alpha, test_length
        )
        # for test add train_length to each index
        for client_id in range(len(partition_result_test)):
            for index in range(len(partition_result_test[client_id])):
                partition_result_test[client_id][index] += len(train_index_list)

        print("store data in h5 data")
        partition = h5py.File(args.partition_file, "a")

        flag_str = "label" if args.skew_type == "label" else "cluster"
        # delete the old partition files in h5 so that we can write to  the h5 file
        if (
            "/niid_"
            + flag_str
            + "_clients=%.1f_alpha=%g" % (args.client_number, args.alpha)
            in partition
        ):
            del partition[
                "/niid_"
                + flag_str
                + "_clients=%d_alpha=%g" % (args.client_number, args.alpha)
            ]
        if (
            "/niid_"
            + flag_str
            + "_clients=%d_alpha=%g" % (args.client_number, args.alpha)
            in partition
        ):
            del partition[
                "/niid_"
                + flag_str
                + "_clients=%d_alpha=%g" % (args.client_number, args.alpha)
            ]

        partition[
            "/niid_"
            + flag_str
            + "_clients=%d_alpha=%g" % (args.client_number, args.alpha)
            + "/n_clients"
        ] = client_num
        partition[
            "/niid_"
            + flag_str
            + "_clients=%d_alpha=%g" % (args.client_number, args.alpha)
            + "/alpha"
        ] = alpha
        for partition_id in range(client_num):
            train = partition_result_train[partition_id]
            test = partition_result_test[partition_id]
            train_path = (
                "/niid_"
                + flag_str
                + "_clients=%d_alpha=%g" % (args.client_number, args.alpha)
                + "/partition_data/"
                + str(partition_id)
                + "/train/"
            )
            test_path = (
                "/niid_"
                + flag_str
                + "_clients=%d_alpha=%g" % (args.client_number, args.alpha)
                + "/partition_data/"
                + str(partition_id)
                + "/test/"
            )
            partition[train_path] = train
            partition[test_path] = test
        partition.close()
    else:
        print("store data in h5 data")
        partition = h5py.File(args.partition_file, "a")
        partition[
            "/natural" + "_clients=%d" % (args.client_number) + "/n_clients"
        ] = client_num
        partition[
            "/natural" + "_clients=%d" % (args.client_number) + "/natural_factors"
        ] = len(label_vocab)
        for partition_id in range(client_num):
            train = partition_result_train[partition_id]
            test = partition_result_test[partition_id]
            train_path = (
                "/natural"
                + "_clients=%d" % (args.client_number)
                + "/partition_data/"
                + str(partition_id)
                + "/train/"
            )
            test_path = (
                "/natural"
                + "_clients=%d" % (args.client_number)
                + "/partition_data/"
                + str(partition_id)
                + "/test/"
            )
            partition[train_path] = train
            partition[test_path] = test
        partition.close()


main()
