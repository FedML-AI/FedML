# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import os.path

import h5py
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms


class DatasetHDF5(data.Dataset):
    def __init__(self, hdf5fn, t, transform=None, target_transform=None):
        """
        t: 'train' or 'val'
        """
        super(DatasetHDF5, self).__init__()
        self.hf = h5py.File(hdf5fn, "r", libver="latest", swmr=True)
        self.t = t
        self.n_images = self.hf["%s_img" % self.t].shape[0]
        self.dlabel = self.hf["%s_labels" % self.t][...]
        self.d = self.hf["%s_img" % self.t]
        # self.transform = transform
        # self.target_transform = target_transform

    def _get_dataset_x_and_target(self, index):
        img = self.d[index, ...]
        target = self.dlabel[index]
        return img, np.int64(target)

    def __getitem__(self, index):
        img, target = self._get_dataset_x_and_target(index)
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return img, target

    def __len__(self):
        return self.n_images


class ImageNet_hdf5(data.Dataset):
    def __init__(
        self,
        data_dir,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        """
        Generating this class too many times will be time-consuming.
        So it will be better calling this once and put it into ImageNet_truncated.
        """
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.hdf5fn = os.path.join(data_dir)

        # if self.train:
        #     self.data_dir = os.path.join(data_dir, 'train')
        # else:
        #     self.data_dir = os.path.join(data_dir, 'val')

        self.all_data_hdf5 = DatasetHDF5(
            self.hdf5fn,
            "train" if self.train else "val",
            transform=self.transform,
            target_transform=self.target_transform,
        )

        self.data_local_num_dict, self.net_dataidx_map = self._get_net_dataidx_map()

        """
            self.local_data_idx is a list containing indexes of local client
        """
        self.all_data_idx = range(len(self.all_data_hdf5))
        if dataidxs == None:
            self.local_data_idx = self.all_data_idx
        elif type(dataidxs) == int:
            self.local_data_idx = self.net_dataidx_map[dataidxs]
        else:
            self.local_data_idx = []
            for idxs in dataidxs:
                self.local_data_idx += self.net_dataidx_map[idxs]

    def _get_net_dataidx_map(self):
        data_local_num_dict = dict()
        net_dataidx_map = dict()
        for i, label in enumerate(self.all_data_hdf5.dlabel):
            label_int = np.int64(label)
            if label in net_dataidx_map:
                net_dataidx_map[label_int].append(i)
            else:
                net_dataidx_map[label_int] = []
                net_dataidx_map[label_int].append(i)

        for key, value in net_dataidx_map.items():
            data_local_num_dict[key] = len(value)

        return data_local_num_dict, net_dataidx_map

    def get_net_dataidx_map(self):
        return self.net_dataidx_map

    def get_data_local_num_dict(self):
        return self.data_local_num_dict

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.all_data_hdf5[self.local_data_idx[index]]
        img = transforms.ToPILImage()(img)
        # img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.local_data_idx)


class ImageNet_truncated_hdf5(data.Dataset):
    def __init__(
        self,
        imagenet_dataset: ImageNet_hdf5,
        dataidxs,
        net_dataidx_map,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):

        self.dataidxs = dataidxs
        self.train = train
        # self.transform = transform
        # self.target_transform = target_transform
        self.download = download

        self.all_data_hdf5 = imagenet_dataset

        self.data_local_num_dict = imagenet_dataset.data_local_num_dict

        self.net_dataidx_map = imagenet_dataset.net_dataidx_map

        """
            self.local_data_idx is a list containing indexes of local client
        """
        self.all_data_idx = range(len(self.all_data_hdf5))
        if dataidxs == None:
            self.local_data_idx = self.all_data_idx
        elif type(dataidxs) == int:
            self.local_data_idx = self.net_dataidx_map[dataidxs]
        else:
            self.local_data_idx = []
            for idxs in dataidxs:
                self.local_data_idx += self.net_dataidx_map[idxs]

    def get_net_dataidx_map(self):
        return self.net_dataidx_map

    def get_data_local_num_dict(self):
        return self.data_local_num_dict

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        # Transform operation has been conducted in all_data_hdf5
        img, target = self.all_data_hdf5[self.local_data_idx[index]]
        return img, target

    def __len__(self):
        return len(self.local_data_idx)
