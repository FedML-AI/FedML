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
    # This implies that we are also dropping the rows whe    # This is in line with Pyvertical but a potential issue
    df_medication.dropna(axis=0,how='any',thresh=None,subset=None,inplace=True)
    df_medication_new = df_medication
    return df_medication_new

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

def time_conversion(time):
    time_list = []
    for index, value in time.items():
        time_list.append(datetime.strptime(time[index], "%Y-%m-%dT%H:%M:%SZ"))
    return time_list

def preprocessing_df(medication_df):
    medication_df_num = medication_df.select_dtypes(exclude='object')
    medication_df_str = medication_df.select_dtypes(include='object')
    for name, input in medication_df_str.items():
        if name == 'STOP':
            print("retreiving", name)
            stop = time_conversion(medication_df_str[name])
            print("stopped retreiving", name)
        elif name == 'START':
            print("retreiving", name)
            start = time_conversion(medication_df_str[name])
            print("stopped retreiving", name)
        # ToDo if I run this with the whole data set I get a zsh kill
        # elif name == 'PATIENT':
        #     print("OneHotEncode", name)
        #     encoded_columns = pd.get_dummies(medication_df_str[name])
        #     medication_df_str = medication_df_str.join(encoded_columns).drop(name, axis=1)
        #     print("Stopped", name)
            
    time_dif = []
    for i in range(len(stop)):
        time_dif.append(((stop[i]-start[i]).total_seconds())/(60*24))
        
    medication_df_num['TIME_DIF'] = time_dif
    medication_df_num_norm = normalize_df(medication_df_num)
    medication_df_num_norm.reset_index(inplace = True)

    # medication_df_str = medication_df_str.drop(['START'], axis=1)
    # medication_df_str = medication_df_str.drop(['STOP'], axis=1)
    # medication_df_str.reset_index(inplace = True)
    # medication_df_preprocessed = pd.concat([medication_df_num_norm, medication_df_str], axis=1 )
    # return medication_df_preprocessed
    return medication_df_num_norm

def process_data(medication_df):
    medication_feat_df = medication_df.copy()
    medication_feat_df.pop('CODE')
    print("[INFO] Extracted the prepare target CODE of the medication data.")
    medication_target_df = medication_df[['CODE']]
    # medication_target_df = pd.get_dummies(medication_target_df)
    medication_target_df.reset_index(inplace=True)
    # (1) Dropped as it is more or less the same as CODE (2) as Reason rest issue of zsh kill
    medication_feat_df = medication_feat_df.drop(['DESCRIPTION','REASONDESCRIPTION','PAYER','ENCOUNTER','PATIENT'], axis=1) #
    norm_medication_feat_df = preprocessing_df(medication_feat_df)
    print(norm_medication_feat_df.head())
    processed_medication_df = pd.concat([norm_medication_feat_df, medication_target_df], axis=1) #, ignore_index= True)
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


def synthea_load_two_party_data(data_dir):
    print("[INFO] load two party data")
    processed_synthea_df = load_processed_data(data_dir)
    print(processed_synthea_df.head())
    party_a_feat_list = ['BASE_COST','PAYER_COVERAGE','TIME_DIF']
    party_b_feat_list = ['DISPENSES','TOTALCOST','REASONCODE']
    Xa, Xb, y = processed_synthea_df[party_a_feat_list].values, processed_synthea_df[party_b_feat_list].values, \
                processed_synthea_df['CODE'].values
    y = np.expand_dims(y, axis=1)
    print("Before",y.shape)
    n_train = int(0.8 * Xa.shape[0])
    print("# of train samples:", n_train)
    Xa_train, Xb_train = Xa[:n_train], Xb[:n_train]
    Xa_test, Xb_test = Xa[n_train:], Xb[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    return [Xa_train, Xb_train, y_train], [Xa_test, Xb_test, y_test]


if __name__ == '__main__':
    data_dir = "../../../data/synthea_medication/"
    synthea_load_two_party_data(data_dir)