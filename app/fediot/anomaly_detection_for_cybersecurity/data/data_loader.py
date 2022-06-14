import os

import numpy as np
import pandas as pd
import torch


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
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_num = 0
    test_data_num = 0
    min_dataset = np.loadtxt(os.path.join(args.data_cache_dir, "min_dataset.txt"))
    max_dataset = np.loadtxt(os.path.join(args.data_cache_dir, "max_dataset.txt"))
    for i, device in enumerate(device_list):
        benign_data = pd.read_csv(
            os.path.join(args.data_cache_dir, device, "benign_traffic.csv")
        )
        benign_data = benign_data[:5000]
        benign_data = np.array(benign_data)
        benign_data[np.isnan(benign_data)] = 0
        benign_data = (benign_data - min_dataset) / (max_dataset - min_dataset)

        g_attack_data_list = [
            os.path.join(args.data_cache_dir, device, "gafgyt_attacks", f)
            for f in os.listdir(os.path.join(args.data_cache_dir, device, "gafgyt_attacks"))
        ]
        if device == "Ennio_Doorbell" or device == "Samsung_SNH_1011_N_Webcam":
            attack_data_list = g_attack_data_list
        else:
            m_attack_data_list = [
                os.path.join(args.data_cache_dir, device, "mirai_attacks", f)
                for f in os.listdir(
                    os.path.join(args.data_cache_dir, device, "mirai_attacks")
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
