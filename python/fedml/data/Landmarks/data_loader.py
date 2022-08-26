import collections
import csv
import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import Landmarks


def _read_csv(path: str):
    """Reads a csv file, and returns the content inside a list of dictionaries.
    Args:
      path: The path to the csv file.
    Returns:
      A list of dictionaries. Each row in the csv file will be a list entry. The
      dictionary is keyed by the column names.
    """
    with open(path, "r") as f:
        return list(csv.DictReader(f))


# class Cutout(object):
#     def __init__(self, length):
#         self.length = length

#     def __call__(self, img):
#         h, w = img.size(1), img.size(2)
#         mask = np.ones((h, w), np.float32)
#         y = np.random.randint(h)
#         x = np.random.randint(w)

#         y1 = np.clip(y - self.length // 2, 0, h)
#         y2 = np.clip(y + self.length // 2, 0, h)
#         x1 = np.clip(x - self.length // 2, 0, w)
#         x2 = np.clip(x + self.length // 2, 0, w)

#         mask[y1: y2, x1: x2] = 0.
#         mask = torch.from_numpy(mask)
#         mask = mask.expand_as(img)
#         img *= mask
#         return img

# def _data_transforms_landmarks():
#     landmarks_MEAN = [0.5071, 0.4865, 0.4409]
#     landmarks_STD = [0.2673, 0.2564, 0.2762]

#     train_transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(landmarks_MEAN, landmarks_STD),
#     ])

#     train_transform.transforms.append(Cutout(16))

#     valid_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(landmarks_MEAN, landmarks_STD),
#     ])

#     return train_transform, valid_transform


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_landmarks():
    # IMAGENET_MEAN = [0.5071, 0.4865, 0.4409]
    # IMAGENET_STD = [0.2673, 0.2564, 0.2762]

    IMAGENET_MEAN = [0.5, 0.5, 0.5]
    IMAGENET_STD = [0.5, 0.5, 0.5]

    image_size = 224
    train_transform = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, valid_transform


def get_mapping_per_user(fn):
    """
    mapping_per_user is {'user_id': [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ... {}],
                         'user_id': [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ... {}],
    } or
                        [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ...
                         {'user_id': xxx, 'image_id': xxx, 'class': xxx} ... ]
    }
    """
    mapping_table = _read_csv(fn)
    expected_cols = ["user_id", "image_id", "class"]
    if not all(col in mapping_table[0].keys() for col in expected_cols):
        logging.error("%s has wrong format.", mapping_table)
        raise ValueError(
            "The mapping file must contain user_id, image_id and class columns. "
            "The existing columns are %s" % ",".join(mapping_table[0].keys())
        )

    data_local_num_dict = dict()

    mapping_per_user = collections.defaultdict(list)
    data_files = []
    net_dataidx_map = {}
    sum_temp = 0

    for row in mapping_table:
        user_id = row["user_id"]
        mapping_per_user[user_id].append(row)
    for user_id, data in mapping_per_user.items():
        num_local = len(mapping_per_user[user_id])
        # net_dataidx_map[user_id]= (sum_temp, sum_temp+num_local)
        # data_local_num_dict[user_id] = num_local
        net_dataidx_map[int(user_id)] = (sum_temp, sum_temp + num_local)
        data_local_num_dict[int(user_id)] = num_local
        sum_temp += num_local
        data_files += mapping_per_user[user_id]
    assert sum_temp == len(data_files)

    return data_files, data_local_num_dict, net_dataidx_map


# for centralized training
def get_dataloader(
    dataset, datadir, train_files, test_files, train_bs, test_bs, dataidxs=None
):
    return get_dataloader_Landmarks(
        datadir, train_files, test_files, train_bs, test_bs, dataidxs
    )


# for local devices
def get_dataloader_test(
    dataset,
    datadir,
    train_files,
    test_files,
    train_bs,
    test_bs,
    dataidxs_train,
    dataidxs_test,
):
    return get_dataloader_test_Landmarks(
        datadir,
        train_files,
        test_files,
        train_bs,
        test_bs,
        dataidxs_train,
        dataidxs_test,
    )


def get_dataloader_Landmarks(
    datadir, train_files, test_files, train_bs, test_bs, dataidxs=None
):
    dl_obj = Landmarks

    transform_train, transform_test = _data_transforms_landmarks()

    train_ds = dl_obj(
        datadir,
        train_files,
        dataidxs=dataidxs,
        train=True,
        transform=transform_train,
        download=True,
    )
    test_ds = dl_obj(
        datadir,
        test_files,
        dataidxs=None,
        train=False,
        transform=transform_test,
        download=True,
    )

    train_dl = data.DataLoader(
        dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False
    )
    test_dl = data.DataLoader(
        dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False
    )

    return train_dl, test_dl


def get_dataloader_test_Landmarks(
    datadir,
    train_files,
    test_files,
    train_bs,
    test_bs,
    dataidxs_train=None,
    dataidxs_test=None,
):
    dl_obj = Landmarks

    transform_train, transform_test = _data_transforms_landmarks()

    train_ds = dl_obj(
        datadir,
        train_files,
        dataidxs=dataidxs_train,
        train=True,
        transform=transform_train,
        download=True,
    )
    test_ds = dl_obj(
        datadir,
        test_files,
        dataidxs=dataidxs_test,
        train=False,
        transform=transform_test,
        download=True,
    )

    train_dl = data.DataLoader(
        dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False
    )
    test_dl = data.DataLoader(
        dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False
    )

    return train_dl, test_dl


def load_partition_data_landmarks(
    dataset,
    data_dir,
    fed_train_map_file,
    fed_test_map_file,
    partition_method=None,
    partition_alpha=None,
    client_number=233,
    batch_size=10,
):

    train_files, data_local_num_dict, net_dataidx_map = get_mapping_per_user(
        fed_train_map_file
    )
    test_files = _read_csv(fed_test_map_file)

    class_num = len(np.unique([item["class"] for item in train_files]))
    # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = len(train_files)

    train_data_global, test_data_global = get_dataloader(
        dataset, data_dir, train_files, test_files, batch_size, batch_size
    )
    # logging.info("train_dl_global number = " + str(len(train_data_global)))
    # logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_files)

    # get local dataset
    data_local_num_dict = data_local_num_dict
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        # local_data_num = len(dataidxs)
        local_data_num = dataidxs[1] - dataidxs[0]
        # data_local_num_dict[client_idx] = local_data_num
        # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(
            dataset, data_dir, train_files, test_files, batch_size, batch_size, dataidxs
        )
        # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        #     client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    # logging("data_local_num_dict: %s" % data_local_num_dict)
    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )


if __name__ == "__main__":
    data_dir = "./cache/images"
    fed_g23k_train_map_file = (
        "../../../data/gld/data_user_dict/gld23k_user_dict_train.csv"
    )
    fed_g23k_test_map_file = (
        "../../../data/gld/data_user_dict/gld23k_user_dict_test.csv"
    )

    fed_g160k_train_map_file = (
        "../../../data/gld/data_user_dict/gld160k_user_dict_train.csv"
    )
    fed_g160k_map_file = "../../../data/gld/data_user_dict/gld160k_user_dict_test.csv"

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

    for client_idx in range(client_number):
        i = 0
        for data, label in train_data_local_dict[client_idx]:
            print(data)
            print(label)
            i += 1
            if i > 5:
                break
