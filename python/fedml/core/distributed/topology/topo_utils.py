import math
import numpy as np
import networkx as nx


def get_2d_torus_overlay(node_num):
    side_len = node_num ** 0.5
    assert math.ceil(side_len) == math.floor(side_len)
    side_len = int(side_len)

    torus = np.zeros((node_num, node_num), dtype=np.float32)

    for i in range(side_len):
        for j in range(side_len):
            idx = i * side_len + j
            torus[i, i] = 1 / 5
            torus[idx, (((i + 1) % side_len) * side_len + j)] = 1 / 5
            torus[idx, (((i - 1) % side_len) * side_len + j)] = 1 / 5
            torus[idx, (i * side_len + (j + 1) % side_len)] = 1 / 5
            torus[idx, (i * side_len + (j - 1) % side_len)] = 1 / 5

    return torus


def get_star_overlay(node_num):

    star = np.zeros((node_num, node_num), dtype=np.float32)
    for i in range(node_num):
        if i == 0:
            star[i, i] = 1 / node_num
        else:
            star[0, i] = star[i, 0] = 1 / node_num
            star[i, i] = 1 - 1 / node_num

    return star


def get_complete_overlay(node_num):

    complete = np.ones((node_num, node_num), dtype=np.float32)
    complete /= node_num

    return complete


def get_isolated_overlay(node_num):

    isolated = np.zeros((node_num, node_num), dtype=np.float32)

    for i in range(node_num):
        isolated[i, i] = 1

    return isolated


def get_balanced_tree_overlay(node_num, degree=2):

    tree = np.zeros((node_num, node_num), dtype=np.float32)

    for i in range(node_num):
        for j in range(1, degree+1):
            k = i * 2 + j
            if k >= node_num:
                break
            tree[i, k] = 1 / (degree+1)

    for i in range(node_num):
        tree[i, i] = 1 - tree[i, :].sum()

    return tree


def get_barbell_overlay(node_num, m1=1, m2=0):

    barbell = None

    return barbell


def get_random_overlay(node_num, probability=0.5):

    random = np.array(
        nx.to_numpy_matrix(nx.fast_gnp_random_graph(node_num, probability)), dtype=np.float32
    )

    matrix_sum = random.sum(1)

    for i in range(node_num):
        for j in range(node_num):
            if i != j and random[i, j] > 0:
                random[i, j] = 1 / (1 + max(matrix_sum[i], matrix_sum[j]))
        random[i, i] = 1 - random[i].sum()

    return random