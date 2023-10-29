import logging
import os
import torch
import flamby

from flamby.datasets.fed_ixi.dataset import FedIXITiny
from monai.transforms import Compose, NormalizeIntensity

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from flamby.utils import accept_license, create_config, write_value_in_config

# IXI Tiny

TINY_URL = (
    "https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/7kd5wj7v7p-1.zip"
)


def dl_ixi_tiny(output_folder, debug=False):
    """
    Download the IXI Tiny dataset.

    Parameters
        ----------
        output_folder : str
            The folder where to download the dataset.
    """
    print(
        "The IXI dataset is made available under the Creative Commons CC BY-SA 3.0 license.\n\
    If you use the IXI data please acknowledge the source of the IXI data, e.g. the following website: https://brain-development.org/ixi-dataset/\n\
    IXI Tiny is derived from the same source. Acknowledge the following reference on TorchIO : https://torchio.readthedocs.io/datasets.html#ixitiny\n\
    Pérez-García F, Sparks R, Ourselin S. TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning. arXiv:2003.04696 [cs, eess, stat]. 2020. https://doi.org/10.48550/arXiv.2003.04696"
    )
    print("IN FUNCTION")
    print(output_folder)
    accept_license("https://brain-development.org/ixi-dataset/", "fed_ixi")
    os.makedirs(output_folder, exist_ok=True)

    # Creating config file with path to dataset
    dict, config_file = create_config(output_folder, debug, "fed_ixi")
    print("Path to confing file : " +str(config_file))
    if dict["download_complete"]:
        print("You have already downloaded the IXI dataset, aborting.")
        sys.exit()

    img_zip_archive_name = TINY_URL.split("/")[-1]
    img_archive_path = Path(output_folder).joinpath(img_zip_archive_name)

    with requests.get(TINY_URL, stream=True) as response:
        # Raise error if not 200
        response.raise_for_status()
        file_size = int(response.headers.get("Content-Length", 0))
        desc = "(Unknown total file size)" if file_size == 0 else ""
        print(f"Downloading to {img_archive_path}")
        with tqdm.wrapattr(response.raw, "read", total=file_size, desc=desc) as r_raw:
            with open(img_archive_path, "wb") as f:
                shutil.copyfileobj(r_raw, f)

    print("********************Extraction Started****************************")
    # extraction
    print(f"Extracting to {output_folder}")
    with zipfile.ZipFile(f"{img_archive_path}", "r") as zip_ref:
        for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
            zip_ref.extract(member=file, path=output_folder)

    print("********************Extraction Complete****************************")
    write_value_in_config(config_file, "download_complete", True)
    write_value_in_config(config_file, "preprocessing_complete", True)
    print("********************Write Value Config File Complete****************************")

def load_partition_fed_ixi(args):

    if args.download:
        #from flamby.datasets.fed_ixi.dataset_creation_scripts.download import (
        #    main as download_main,
        #)
        print(args.data_cache_dir)
        if (not os.path.exists(args.data_cache_dir)) or len(
            os.listdir(args.data_cache_dir)
        ) == 0:
        #    download_main(args.data_cache_dir, args.debug)
            dl_ixi_tiny(args.data_cache_dir, args.debug)

    train_data_num = 0
    test_data_num = 0
    train_data_global = None
    test_data_global = None
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    nc = 2

    if args.process_id == 0:  # server
        pass
    else:  # client
        logging.info(f"load center {int(args.process_id)-1} data")
        client_idx = int(args.process_id) - 1
        print("*************************")
        print(client_idx)
        print("*************************")
        training_transform = Compose([
            NormalizeIntensity(),
        ])
        test_transform = Compose([
            NormalizeIntensity(),
        ])
        train_dataset = FedIXITiny(
            transform = training_transform, center=client_idx, train=True, pooled=True, debug=args.debug
        ) #might need change [check ]
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.worker_num,
        )
        test_dataset = FedIXITiny(
            transform = test_transform, center=client_idx, train=False, pooled=True, debug=args.debug
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.worker_num,
            drop_last=True,
        )
        data_local_num_dict[client_idx] = len(train_dataset)
        train_data_local_dict[client_idx] = train_dataloader
        test_data_local_dict[client_idx] = test_dataloader

    # logging.info(f"train_data_num: {train_data_num}")
    # logging.info(f"test_data_num: {test_data_num}")
    # logging.info(f"train_data_global: {train_data_global}")
    # logging.info(f"test_data_global: {test_data_global}")
    # logging.info(f"data_local_num_dict: {data_local_num_dict}")
    # logging.info(f"train_data_local_dict: {train_data_local_dict}")
    # logging.info(f"test_data_local_dict: {test_data_local_dict}")
    # logging.info(f"nc: {nc}")

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        nc,
    )
