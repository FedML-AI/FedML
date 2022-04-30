import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .lending_club_feature_group import (
    all_feature_list,
    qualification_feat,
    loan_feat,
    debt_feat,
    repayment_feat,
    multi_acc_feat,
    mal_behavior_feat,
)

target_map = {"Good Loan": 0, "Bad Loan": 1}

grade_map = {"A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1, "G": 0}
sub_grade = [
    "C1"
    "D2"
    "D1"
    "C4"
    "C3"
    "C2"
    "D5"
    "B3"
    "A4"
    "B5"
    "C5"
    "D4"
    "E1"
    "E4"
    "B4"
    "D3"
    "A1"
    "E5"
    "B2"
    "B1"
    "A5"
    "F5"
    "A3"
    "E3"
    "A2"
    "E2"
    "F4"
    "G1"
    "G2"
    "F1"
    "F2"
    "F3"
    "G4"
    "G3"
    "G5"
]
emp_length_map = {
    np.nan: 0,
    "< 1 year": 1,
    "1 year": 2,
    "2 years": 2,
    "3 years": 2,
    "4 years": 3,
    "5 years": 3,
    "6 years": 3,
    "7 years": 4,
    "8 years": 4,
    "9 years": 4,
    "10+ years": 5,
}

home_ownership_map = {
    "RENT": 0,
    "MORTGAGE": 1,
    "OWN": 2,
    "ANY": 3,
    "NONE": 3,
    "OTHER": 3,
}

verification_status_map = {"Not Verified": 0, "Source Verified": 1, "Verified": 2}

# verification_status_joint = [nan 'Verified' 'Not Verified' 'Source Verified']

term_map = {" 36 months": 0, " 60 months": 1}
initial_list_status_map = {"w": 0, "f": 1}
purpose_map = {
    "debt_consolidation": 0,
    "credit_card": 0,
    "small_business": 1,
    "educational": 2,
    "car": 3,
    "other": 3,
    "vacation": 3,
    "house": 3,
    "home_improvement": 3,
    "major_purchase": 3,
    "medical": 3,
    "renewable_energy": 3,
    "moving": 3,
    "wedding": 3,
}
application_type_map = {"Individual": 0, "Joint App": 1}
disbursement_method_map = {"Cash": 0, "DirectPay": 1}


def normalize(x):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled


def normalize_df(df):
    column_names = df.columns
    x = df.values
    x_scaled = normalize(x)
    scaled_df = pd.DataFrame(data=x_scaled, columns=column_names)
    return scaled_df


def loan_condition(status):
    bad_loan = [
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
        "In Grace Period",
        "Late (16-30 days)",
        "Late (31-120 days)",
    ]
    if status in bad_loan:
        return "Bad Loan"
    else:
        return "Good Loan"


def compute_annual_income(row):
    if row["verification_status"] == row["verification_status_joint"]:
        return row["annual_inc_joint"]
    return row["annual_inc"]


def determine_good_bad_loan(df_loan):
    print("[INFO] determine good or bad loan")

    df_loan["target"] = np.nan
    df_loan["target"] = df_loan["loan_status"].apply(loan_condition)
    return df_loan


def determine_annual_income(df_loan):
    print("[INFO] determine annual income")

    df_loan["annual_inc_comp"] = np.nan
    df_loan["annual_inc_comp"] = df_loan.apply(compute_annual_income, axis=1)
    return df_loan


def determine_issue_year(df_loan):
    print("[INFO] determine issue year")

    # transform the issue dates by year
    dt_series = pd.to_datetime(df_loan["issue_d"])
    df_loan["issue_year"] = dt_series.dt.year
    return df_loan


def digitize_columns(data_frame):
    print("[INFO] digitize columns")

    data_frame = data_frame.replace(
        {
            "target": target_map,
            "grade": grade_map,
            "emp_length": emp_length_map,
            "home_ownership": home_ownership_map,
            "verification_status": verification_status_map,
            "term": term_map,
            "initial_list_status": initial_list_status_map,
            "purpose": purpose_map,
            "application_type": application_type_map,
            "disbursement_method": disbursement_method_map,
        }
    )
    return data_frame


def prepare_data(file_path):
    print("[INFO] prepare loan data.")

    df_loan = pd.read_csv(file_path, low_memory=False)
    # print(f"[INFO] loaded loan data with shape:{df_loan.shape} to :{file_path}")

    df_loan = determine_good_bad_loan(df_loan)
    df_loan = determine_annual_income(df_loan)
    df_loan = determine_issue_year(df_loan)
    df_loan = digitize_columns(df_loan)

    df_loan = df_loan[df_loan["issue_year"] == 2018]
    return df_loan


def process_data(loan_df):
    loan_feat_df = loan_df[all_feature_list]
    loan_feat_df = loan_feat_df.fillna(-99)
    assert loan_feat_df.isnull().sum().sum() == 0

    norm_loan_feat_df = normalize_df(loan_feat_df)
    loan_target_df = loan_df[["target"]]
    processed_loan_df = pd.concat([norm_loan_feat_df, loan_target_df], axis=1)
    return processed_loan_df


def load_processed_data(data_dir):
    file_path = data_dir + "processed_loan.csv"
    if os.path.exists(file_path):
        print(f"[INFO] load processed loan data from {file_path}")
        processed_loan_df = pd.read_csv(file_path, low_memory=False)
    else:
        # print(f"[INFO] start processing loan data.")
        file_path = data_dir + "loan.csv"
        processed_loan_df = process_data(prepare_data(file_path))
        file_path = data_dir + "processed_loan.csv"
        processed_loan_df.to_csv(file_path, index=False)
        print(f"[INFO] save processed loan data to: {file_path}")
    return processed_loan_df


def loan_load_two_party_data(data_dir):
    print("[INFO] load two party data")
    processed_loan_df = load_processed_data(data_dir)
    party_a_feat_list = qualification_feat + loan_feat
    party_b_feat_list = debt_feat + repayment_feat + multi_acc_feat + mal_behavior_feat
    Xa, Xb, y = (
        processed_loan_df[party_a_feat_list].values,
        processed_loan_df[party_b_feat_list].values,
        processed_loan_df["target"].values,
    )

    y = np.expand_dims(y, axis=1)
    n_train = int(0.8 * Xa.shape[0])
    print("# of train samples:", n_train)
    Xa_train, Xb_train = Xa[:n_train], Xb[:n_train]
    Xa_test, Xb_test = Xa[n_train:], Xb[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print("Xa_train.shape:", Xa_train.shape)
    print("Xb_train.shape:", Xb_train.shape)
    print("Xa_test.shape:", Xa_test.shape)
    print("Xb_test.shape:", Xb_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape, type(y_test))
    return [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test]


def loan_load_three_party_data(data_dir):
    print("[INFO] load three party data")
    processed_loan_df = load_processed_data(data_dir)
    party_a_feat_list = qualification_feat + loan_feat
    party_b_feat_list = debt_feat + repayment_feat
    party_c_feat_list = multi_acc_feat + mal_behavior_feat
    Xa, Xb, Xc, y = (
        processed_loan_df[party_a_feat_list].values,
        processed_loan_df[party_b_feat_list].values,
        processed_loan_df[party_c_feat_list].values,
        processed_loan_df["target"].values,
    )

    y = np.expand_dims(y, axis=1)
    n_train = int(0.8 * Xa.shape[0])
    Xa_train, Xb_train, Xc_train = Xa[:n_train], Xb[:n_train], Xc[:n_train]
    Xa_test, Xb_test, Xc_test = Xa[n_train:], Xb[n_train:], Xc[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print("Xa_train.shape:", Xa_train.shape)
    print("Xb_train.shape:", Xb_train.shape)
    print("Xc_train.shape:", Xc_train.shape)
    print("Xa_test.shape:", Xa_test.shape)
    print("Xb_test.shape:", Xb_test.shape)
    print("Xc_test.shape:", Xc_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape)
    return [Xa_train, Xb_train, Xc_train, y_train], [Xa_test, Xb_test, Xc_test, y_test]


if __name__ == "__main__":
    data_dir = "../../../data/lending_club_loan/"
    loan_load_two_party_data(data_dir)
