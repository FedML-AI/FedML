from .data_loader import load_partition_data_landmarks

"""
    You can run with python check_download.py to check if you have all 
    data samples in federated_train.csv and test.csv.
"""

if __name__ == "__main__":
    data_dir = "./cache/images"
    fed_g23k_train_map_file = "./cache/datasets/mini_gld_train_split.csv"
    fed_g23k_test_map_file = "./cache/datasets/mini_gld_test.csv"

    fed_g160k_train_map_file = (
        "./cache/datasets/landmarks-user-160k/federated_train.csv"
    )
    fed_g160k_map_file = "./cache/datasets/landmarks-user-160k/test.csv"

    dataset_name = "g160k"

    if dataset_name == "g23k":
        client_number = 233
        fed_train_map_file = fed_g23k_train_map_file
        fed_test_map_file = fed_g23k_test_map_file
    elif dataset_name == "g160k":
        client_number = 1262
        fed_train_map_file = fed_g160k_train_map_file
        fed_test_map_file = fed_g160k_map_file

    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_partition_data_landmarks(
        None,
        data_dir,
        fed_train_map_file,
        fed_test_map_file,
        partition_method=None,
        partition_alpha=None,
        client_number=client_number,
        batch_size=10,
    )

    print(train_data_num, test_data_num, class_num)
    print(data_local_num_dict)

    i = 0
    for data, label in train_data_global:
        print(data)
        print(label)
        i += 1
        if i > 5:
            break
    print("=============================\n")

    flag = True
    for client_idx in range(client_number):
        for i, (data, label) in enumerate(train_data_local_dict[client_idx]):
            print("client_idx %d has %s-th data" % (client_idx, i))

    # flag = True
    # for client_idx in range(client_number):
    #     for i, (data, label) in enumerate(test_data_local_dict[client_idx]):
    #         print("client_idx %d has %s-th data" % (client_idx, i))
