import os


def equally_partition_a_dataset(client_num_in_total, dataset):
    client_data_num = int(len(dataset) / client_num_in_total)
    local_data_dict = dict()
    train_data_local_num_dict = dict()
    start_counter = 0
    datasize = len(dataset)
    for i in range(client_num_in_total):
        local_data_dict[i] = dataset[start_counter:start_counter + client_data_num]
        start_counter += client_data_num
        train_data_local_num_dict[i] = client_data_num
    return (
        datasize,
        train_data_local_num_dict,
        local_data_dict,
    )


def read_data(data_dir):
    train_files = os.listdir(data_dir)
    train_files = [f for f in train_files if f.endswith(".txt")]
    dataset = []
    for f in train_files:
        file_path = os.path.join(data_dir, f)
        f2 = open(file_path, "r")
        lines = [int(line.strip()) for line in f2]
        dataset.extend(lines)
    return dataset
