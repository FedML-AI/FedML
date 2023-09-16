import json
import random

import numpy as np
from tqdm import trange

NUM_USER = 30


def softmax(x):
    """
    Compute the softmax values for a given array.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Softmax values.
    """
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex


def generate_synthetic(alpha, beta, iid):
    """
    Generate synthetic data for federated learning.

    Args:
        alpha (float): Mean of user weights.
        beta (float): Mean of user biases.
        iid (int): Unused parameter.

    Returns:
        list: List of user data samples.
        list: List of labels for user data samples.
    """
    dimension = 60
    NUM_CLASS = 10
    np.random.seed(0)
    samples_per_user = np.random.lognormal(4, 2, (NUM_USER)).astype(int) + 50


    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]


    mean_W = np.random.normal(0, alpha, NUM_USER)
    mean_b = mean_W
    B = np.random.normal(0, beta, NUM_USER)
    mean_x = np.zeros((NUM_USER, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(NUM_USER):
        mean_x[i] = np.random.normal(B[i], 1, dimension)


    for i in range(NUM_USER):

        W = np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
        b = np.random.normal(mean_b[i], 1, NUM_CLASS)

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()



    return X_split, y_split


def main():
    train_path = "train/mytrain.json"
    test_path = "test/mytest.json"

    X, y = generate_synthetic(alpha=1, beta=1, iid=0)  # synthetic (1,1)

    # Create data structure
    train_data = {"users": [], "user_data": {}, "num_samples": []}
    test_data = {"users": [], "user_data": {}, "num_samples": []}

    for i in trange(NUM_USER, ncols=120):
        uname = "f_{0:05d}".format(i)
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.9 * num_samples)
        test_len = num_samples - train_len

        train_data["users"].append(uname)
        train_data["user_data"][uname] = {"x": X[i][:train_len], "y": y[i][:train_len]}
        train_data["num_samples"].append(train_len)
        test_data["users"].append(uname)
        test_data["user_data"][uname] = {"x": X[i][train_len:], "y": y[i][train_len:]}
        test_data["num_samples"].append(test_len)

    with open(train_path, "w") as outfile:
        json.dump(train_data, outfile)
    with open(test_path, "w") as outfile:
        json.dump(test_data, outfile)


if __name__ == "__main__":
    main()
