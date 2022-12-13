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
        self.targets = self.hf["%s_labels" % self.t][...]
        self.data = self.hf["%s_img" % self.t]
        # self.dlabel = self.hf["%s_labels" % self.t][...]
        # self.d = self.hf["%s_img" % self.t]
        # self.transform = transform
        # self.target_transform = target_transform

    def _get_dataset_x_and_target(self, index):
        img = self.data[index, ...]
        target = self.targets[index]
        # img = self.d[index, ...]
        # target = self.dlabel[index]
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
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        """
        Generating this class too many times will be time-consuming.
        So it will be better calling this once and put it into ImageNet_truncated.
        """
        hdf5fn = os.path.join(data_dir)
        self.hf = h5py.File(hdf5fn, "r", libver="latest", swmr=True)
        self.t = "train" if train else "val"
        self.n_images = self.hf["%s_img" % self.t].shape[0]
        self.targets = self.hf["%s_labels" % self.t][...]
        self.data = self.hf["%s_img" % self.t]
        self.transform = transform
        self.target_transform = target_transform

    def _get_dataset_x_and_target(self, index):
        img = self.data[index, ...]
        target = self.targets[index]
        return img, np.int64(target)

    def __getitem__(self, index):
        img, target = self._get_dataset_x_and_target(index)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return self.n_images

        
        

class ImageNet_truncated_hdf5(data.Dataset):
    def __init__(
        self,
        imagenet_dataset: ImageNet_hdf5,
        dataidxs=None,
    ):
        self.dataidxs = dataidxs
        self.all_data_hdf5 = imagenet_dataset
        if dataidxs is None:
            self.local_data_idx = list(range(len(self.all_data_hdf5)))
        else:
            self.local_data_idx = dataidxs


    def __getitem__(self, index):
        # Transform operation has been conducted in all_data_hdf5
        img, target = self.all_data_hdf5[self.local_data_idx[index]]
        return img, target

    def __len__(self):
        return len(self.local_data_idx)
