from ......differential_privacy.LDP.frequency_estimation.pure_frequency_oracles.UE import (
    UE,
)
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


def test_sue_client_permute():
    input_data = 2  # real input value
    print("Real value:", input_data)
    ue = UE(attr_domain_size=10, epsilon=1, optimal=False)
    print(
        "Sanitization w/ GRR protocol:", ue.client_permute(input_data)
    )  # k: number of values


def test_sue_server_aggregate():
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
    ue = UE(attr_domain_size=k, epsilon=0.5, optimal=False)
    reports = [ue.client_permute(input_data) for input_data in df["age"]]
    est_freq = ue.server_aggregate(reports)
    mse = mean_squared_error(real_freq, est_freq)
    print(f"est_freq={est_freq}, mse={mse}")


def test_oue_client_permute():
    input_data = 2  # real input value
    print("Real value:", input_data)
    ue = UE(attr_domain_size=10, epsilon=1, optimal=True)
    print(
        "Sanitization w/ GRR protocol:", ue.client_permute(input_data)
    )  # k: number of values


def test_oue_server_aggregate():
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
    ue = UE(attr_domain_size=k, epsilon=0.5, optimal=True)
    reports = [ue.client_permute(input_data) for input_data in df["age"]]
    est_freq = ue.server_aggregate(reports)
    mse = mean_squared_error(real_freq, est_freq)
    print(f"est_freq={est_freq}, mse={mse}")


if __name__ == "__main__":
    test_sue_server_aggregate()
    test_sue_client_permute()
    test_oue_server_aggregate()
    test_oue_client_permute()
