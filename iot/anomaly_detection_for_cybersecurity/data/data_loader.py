import logging
import os
import urllib.request

import numpy as np
import pandas as pd
import torch

def download_data(args, device_name):
    url_root = "https://archive.ics.uci.edu/ml/machine-learning-databases/00442"
    if device_name == "Ennio_Doorbell" or device_name == "Samsung_SNH_1011_N_Webcam":
        file_list = ["benign_traffic.csv", "gafgyt_attacks.rar"]
    else:
        file_list = ["benign_traffic.csv", "gafgyt_attacks.rar", "mirai_attacks.rar"]

    for file_name in file_list:
        url = os.path.join(url_root, device_name, file_name)
        file_saved = os.path.join(args.data_cache_dir, device_name, file_name)
        urllib.request.urlretrieve(url, file_saved)

    os.system(
        "find {} -name '*.rar' -execdir unar {{}} \; -exec rm {{}} \;".format(
            args.data_cache_dir
        )
    )


def load_data(args):
    device_list = [
        "Danmini_Doorbell",
        "Ecobee_Thermostat",
        "Ennio_Doorbell",
        "Philips_B120N10_Baby_Monitor",
        "Provision_PT_737E_Security_Camera",
        "Provision_PT_838_Security_Camera",
        "Samsung_SNH_1011_N_Webcam",
        "SimpleHome_XCS7_1002_WHT_Security_Camera",
        "SimpleHome_XCS7_1003_WHT_Security_Camera",
    ]

    train_data_global = list()
    test_data_global = list()
    train_data_local_dict = dict.fromkeys(range(9))
    test_data_local_dict = dict.fromkeys(range(9))
    train_data_local_num_dict = dict.fromkeys(range(9))
    train_data_num = 0
    test_data_num = 0

    min_dataset = np.loadtxt(os.path.join(args.data_cache_dir, "min_dataset.txt"))
    max_dataset = np.loadtxt(os.path.join(args.data_cache_dir, "max_dataset.txt"))

    if args.rank == 0:
        for i, device_name in enumerate(device_list):
            device_data_cache_dir = os.path.join(args.data_cache_dir, device_name)
            if not os.path.exists(device_data_cache_dir):
                os.makedirs(device_data_cache_dir)
                logging.info("Downloading dataset for all devices on server")
                download_data(args, device_name)

            benign_data = pd.read_csv(
                os.path.join(args.data_cache_dir, device_name, "benign_traffic.csv")
            )
            benign_data = benign_data[:5000]
            benign_data = np.array(benign_data)
            benign_data[np.isnan(benign_data)] = 0
            benign_data = (benign_data - min_dataset) / (max_dataset - min_dataset)

            g_attack_data_list = [
                os.path.join(args.data_cache_dir, device_name, "gafgyt_attacks", f)
                for f in os.listdir(
                    os.path.join(args.data_cache_dir, device_name, "gafgyt_attacks")
                )
            ]
            if (
                device_name == "Ennio_Doorbell"
                or device_name == "Samsung_SNH_1011_N_Webcam"
            ):
                attack_data_list = g_attack_data_list
            else:
                m_attack_data_list = [
                    os.path.join(args.data_cache_dir, device_name, "mirai_attacks", f)
                    for f in os.listdir(
                        os.path.join(args.data_cache_dir, device_name, "mirai_attacks")
                    )
                ]
                attack_data_list = g_attack_data_list + m_attack_data_list

            attack_data = pd.concat([pd.read_csv(f)[:500] for f in attack_data_list])
            attack_data = (attack_data - attack_data.mean()) / (attack_data.std())
            attack_data = np.array(attack_data)
            attack_data[np.isnan(attack_data)] = 0

            train_data_local_dict[i] = torch.utils.data.DataLoader(
                benign_data, batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            test_data_local_dict[i] = torch.utils.data.DataLoader(
                attack_data, batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            train_data_local_num_dict[i] = len(train_data_local_dict[i])
            train_data_num += train_data_local_num_dict[i]

    else:
        device_name = device_list[args.rank - 1]
        device_data_cache_dir = os.path.join(args.data_cache_dir, device_name)
        if not os.path.exists(device_data_cache_dir):
            os.makedirs(device_data_cache_dir)
            logging.info(
                "Downloading dataset for device {} on client".format(args.rank)
            )
            download_data(args, device_name)

        logging.info("Creating dataset {}".format(device_name))
        benign_data = pd.read_csv(
            os.path.join(device_data_cache_dir, "benign_traffic.csv")
        )
        benign_data = benign_data[:5000]
        benign_data = np.array(benign_data)
        benign_data[np.isnan(benign_data)] = 0
        benign_data = (benign_data - min_dataset) / (max_dataset - min_dataset)

        train_data_local_dict[args.rank - 1] = torch.utils.data.DataLoader(
            benign_data, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        train_data_local_num_dict[args.rank - 1] = len(
            train_data_local_dict[args.rank - 1]
        )
        train_data_num += train_data_local_num_dict[args.rank - 1]

    class_num = 115
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num
