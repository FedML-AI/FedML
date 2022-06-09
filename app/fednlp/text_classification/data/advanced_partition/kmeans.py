import h5py
import argparse
import os
import pickle
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


def get_embedding_Kmeans(embedding_exist, corpus, N_clients, bsz=16):
    embedding_data = {}
    corpus_embeddings = []
    if embedding_exist == False:
        embedder = SentenceTransformer(
            "distilbert-base-nli-stsb-mean-tokens", device="cuda:0"
        )  # server only
        corpus_embeddings = embedder.encode(
            corpus, show_progress_bar=True, batch_size=bsz
        )  # smaller batch size for gpu

        embedding_data["data"] = corpus_embeddings
    else:
        corpus_embeddings = corpus
    ### KMEANS clustering
    print("start Kmeans")
    num_clusters = N_clients
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    print("end Kmeans")
    # TODO: read the center points

    return cluster_assignment, embedding_data


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cluster_number",
        type=int,
        default="100",
        metavar="CN",
        help="client number for lda partition",
    )

    parser.add_argument(
        "--bsz",
        type=int,
        default="16",
        metavar="CN",
        help="batch size for sentenceBERT",
    )

    parser.add_argument(
        "--data_file",
        type=str,
        default="data/data_files/wikiner_data.h5",
        metavar="DF",
        help="data pickle file path",
    )

    parser.add_argument(
        "--partition_file",
        type=str,
        default="data/partition_files/wikiner_partition.h5",
        metavar="PF",
        help="partition pickle file path",
    )

    parser.add_argument(
        "--embedding_file",
        type=str,
        default="data/embedding_files/wikiner_embedding.h5",
        metavar="EF",
        help="embedding pickle file path",
    )

    parser.add_argument(
        "--task_type",
        type=str,
        metavar="TT",
        default="text_classfication",
        help="task type",
    )

    parser.add_argument(
        "--overwrite",
        action="store_false",
        default=True,
        help="True if embedding data file does not exist False if it does exist",
    )

    # add a stroe_true for --overwrite
    args = parser.parse_args()
    N_Clients = args.cluster_number
    print("start reading data")
    f = h5py.File(args.data_file, "r")
    attributes = json.loads(f["attributes"][()])
    print(attributes.keys())
    total_index_list = attributes["index_list"]
    test_index_list = []
    train_index_list = []
    if "train_index_list" in attributes:
        test_index_list = attributes["test_index_list"]
        print(len(test_index_list))
        train_index_list = attributes["train_index_list"]
        print(len(train_index_list))

    corpus = []
    if (
        args.task_type == "name_entity_recognition"
    ):  # specifically wnut and wikiner datesets
        for i in f["X"].keys():
            sentence = f["X"][i][()]
            sentence = [i.decode("UTF-8") for i in sentence]
            corpus.append(" ".join(sentence))

    elif args.task_type == "reading_comprehension":  # specifically Squad1.1 dataset
        for i in f["context_X"].keys():
            question_components = []
            # context = f['context_X'][i][()].decode('UTF-8')
            question = f["question_X"][i][()].decode("UTF-8")
            answer_start = f["Y"][i][()][0]
            answer_end = f["Y"][i][()][1]
            answer = f["context_X"][i][()].decode("UTF-8")[answer_start:answer_end]

            question_components.append(question)
            question_components.append(answer)
            corpus.append(" ".join(question_components))

    elif args.task_type == "sequence_to_sequence":
        for i in f["Y"].keys():
            sentence = f["Y"][i][()].decode("UTF-8")
            corpus.append(sentence)

    else:
        for i in f["X"].keys():
            sentence = f["X"][i][()].decode("UTF-8")
            corpus.append(sentence)
    f.close()

    print("start process embedding data and kmeans partition")
    cluster_assignment = []
    embedding_data = []
    if args.overwrite == False:
        cluster_assignment, corpus_embedding = get_embedding_Kmeans(
            False, corpus, N_Clients, args.bsz
        )
        embedding_data = {}
        embedding_data["data"] = corpus_embedding
        with open(args.embedding_file, "wb") as f:
            pickle.dump(embedding_data, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(args.embedding_file, "rb") as f:
            embedding_data = pickle.load(f)
            embedding_data = embedding_data["data"]
            if isinstance(embedding_data, dict):
                embedding_data = embedding_data["data"]
            cluster_assignment, corpus_embedding = get_embedding_Kmeans(
                True, embedding_data, N_Clients, args.bsz
            )

    print("start insert data")
    partition_pkl_train = {}
    partition_pkl_test = {}

    for cluster_id in range(N_Clients):
        partition_pkl_train[cluster_id] = []
        partition_pkl_test[cluster_id] = []

    for index in train_index_list:
        idx = cluster_assignment[index]
        if idx in partition_pkl_train:
            partition_pkl_train[idx].append(index)
        else:
            partition_pkl_train[idx] = [index]

    for index in test_index_list:
        idx = cluster_assignment[index]
        if idx in partition_pkl_test:
            partition_pkl_test[idx].append(index)
        else:
            partition_pkl_test[idx] = [index]

    print("store kmeans partition to file")
    partition = ""
    partition = h5py.File(args.partition_file, "a")
    if "/kmeans_clusters=%d" % args.cluster_number in partition:
        del partition["/kmeans_clusters=%d" % args.cluster_number]

    partition[
        "/kmeans_clusters=%d" % args.cluster_number + "/n_clients"
    ] = args.cluster_number
    partition[
        "/kmeans_clusters=%d" % args.cluster_number + "/client_assignment"
    ] = cluster_assignment

    for i in sorted(partition_pkl_train.keys()):
        train = partition_pkl_train[i]
        test = partition_pkl_test[i]
        train_path = (
            "/kmeans_clusters=%d" % args.cluster_number
            + "/partition_data/"
            + str(i)
            + "/train/"
        )
        test_path = (
            "/kmeans_clusters=%d" % args.cluster_number
            + "/partition_data/"
            + str(i)
            + "/test/"
        )
        partition[train_path] = train
        partition[test_path] = test

    partition.close()


main()
