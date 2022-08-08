from ......differential_privacy.LDP.frequency_estimation.pure_frequency_oracles.LH import (
    LH,
)
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


def test_blh_client_permute():
    input_data = 2  # real input value
    print("Real value:", input_data)
    blh = LH(attr_domain_size=10, epsilon=1, optimal=False)
    print(
        "Sanitization w/ GRR protocol:", blh.client_permute(input_data)
    )  # k: number of values


def test_blh_server_aggregate():
    df = pd.read_csv("../datasets/db_adults.csv", usecols=["age"])
    # ## Encoding values
    LE = LabelEncoder()
    df["age"] = LE.fit_transform(df["age"])

    # ## Static Parameteres
    n = df.shape[0]
    print("Number of Users =", n)

    # attribute's domain size
    k = len(set(df["age"]))
    print("\nAttribute's domain size =", k)

    # Real normalized frequency
    real_freq = np.unique(df, return_counts=True)[-1] / n
    blh = LH(attr_domain_size=k, epsilon=0.5, optimal=False)
    reports = [blh.client_permute(input_data) for input_data in df["age"]]
    est_freq = blh.server_aggregate(reports)
    mse = mean_squared_error(real_freq, est_freq)
    print(f"est_freq={est_freq}, mse={mse}")


def test_olh_client_permute():
    input_data = 2  # real input value
    print("Real value:", input_data)
    olh = LH(attr_domain_size=10, epsilon=1, optimal=True)
    print(
        "Sanitization w/ GRR protocol:", olh.client_permute(input_data)
    )  # k: number of values


def test_olh_server_aggregate():
    df = pd.read_csv("../datasets/db_adults.csv", usecols=["age"])
    # ## Encoding values
    LE = LabelEncoder()
    df["age"] = LE.fit_transform(df["age"])

    # ## Static Parameteres
    n = df.shape[0]
    print("Number of Users =", n)

    # attribute's domain size
    k = len(set(df["age"]))
    print("\nAttribute's domain size =", k)

    # Real normalized frequency
    real_freq = np.unique(df, return_counts=True)[-1] / n
    olh = LH(attr_domain_size=k, epsilon=0.5, optimal=True)
    reports = [olh.client_permute(input_data) for input_data in df["age"]]
    est_freq = olh.server_aggregate(reports)
    mse = mean_squared_error(real_freq, est_freq)
    print(f"est_freq={est_freq}, mse={mse}")


if __name__ == "__main__":
    test_blh_server_aggregate()
    test_blh_client_permute()
    test_olh_server_aggregate()
    test_olh_client_permute()
