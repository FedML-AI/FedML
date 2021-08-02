# This is taken from: Example - Simple training on Synthea Medications dataset through Tensorflow from PyVertical Examples
import os
import pandas as pd
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def prepare_data(file_path):
    print("[INFO] prepare medication data.")
    df_medication = pd.read_csv(file_path, low_memory=False)#, nrows=1000)
    # print(df_medication.head())
    print("before",df_medication.info())
    df_medication.dropna(axis=0,how='any',thresh=None,subset=None,inplace=True)
    print("after",df_medication.info())
    print("[INFO] dropped na from the medication data.")
    return df_medication

def normalize(x):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled

def normalize_df(df):
    print("[INFO] normalizing numeric input")
    column_names = df.columns
    x = df.values
    x_scaled = normalize(x)
    scaled_df = pd.DataFrame(data=x_scaled, columns=column_names)
    return scaled_df

# def map_str_to_int(words):
#     d = dict()
#     unique = 0
#     for word in words:
#         if word not in d:
#             d[word] = unique
#             unique += 1
#     return d,unique
def time_conversion(time):
    time_list = []
    for index, value in time.items():
        time_list.append(datetime.strptime(time[index], "%Y-%m-%dT%H:%M:%SZ"))
    return time_list

def preprocessing_df(medication_df):
    medication_df_num = medication_df.select_dtypes(exclude='object')
    medication_df_num_norm = normalize_df(medication_df_num)
    medication_df_str = medication_df.select_dtypes(include='object')
    # return medication_df_num_norm
    medication_df_str_conv = medication_df_str # ToDo
    for name, input in medication_df_str.items():
        if name == 'STOP' or name == 'START':
            print("replacing",name)
            medication_df_str[name].replace(time_conversion(medication_df_str[name]), inplace=True)
            print("replacing stopped",name)
        # else:
        #     encoded_columns = pd.get_dummies(medication_df_str[name])
        #     medication_df_str_conv = medication_df_str_conv.join(encoded_columns).drop(name, axis=1)

    medication_df_preprocessed = pd.concat([medication_df_num_norm, medication_df_str_conv], axis=1)
    return medication_df_preprocessed



def process_data(medication_df):
    medication_feat_df = medication_df.copy()
    medication_feat_df.pop('CODE')
    print("[INFO] Extracted the prepare target CODE of the medication data.")
    medication_target_df = medication_df[['CODE']]
    norm_medication_feat_df = preprocessing_df(medication_feat_df)
    # print(len(norm_medication_feat_df),"Rows")
    # print(len(norm_medication_feat_df[0]), "Columns")
    processed_medication_df = pd.concat([norm_medication_feat_df, medication_target_df], axis=1)
    return processed_medication_df


def load_processed_data(data_dir):
    file_path = data_dir + "processed_medications.csv"
    if os.path.exists(file_path):
        print(f"[INFO] load processed medications data from {file_path}")
        processed_medication_df = pd.read_csv(file_path, low_memory=False)
    else:
        print(f"[INFO] start processing loan data.")
        file_path = data_dir + "medications.csv"
        processed_medication_df = process_data(prepare_data(file_path))
        file_path = data_dir + "processed_medications.csv"
        processed_medication_df.to_csv(file_path, index=False)
        print(f"[INFO] save processed medications data to: {file_path}")
    return processed_medication_df


def loan_load_two_party_data(data_dir):
    print("[INFO] load two party data")
    processed_loan_df = load_processed_data(data_dir)
    party_a_feat_list = ['BASE_COST','PAYER_COVERAGE']
    party_b_feat_list = ['DISPENSES','TOTALCOST','REASONCODE']
    Xa, Xb, y = processed_loan_df[party_a_feat_list].values, processed_loan_df[party_b_feat_list].values, \
                processed_loan_df['CODE'].values

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



if __name__ == '__main__':
    data_dir = "../../../data/synthea_medication/"
    loan_load_two_party_data(data_dir)