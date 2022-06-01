import os
from abc import ABC
from pathlib import Path, PurePath
from typing import Literal

from torch.utils.data import Dataset

from .utils import _download_file, _extract_file


class COCOBase(Dataset, ABC):
    root_dir: PurePath
    annotations_zip_path: PurePath
    train_zip_path: PurePath
    val_zip_path: PurePath
    test_zip_path: PurePath
    annotations_path: PurePath
    images_path: PurePath
    instances_path: PurePath
    downloaded: bool

    def __init__(self,
                 root_dir: str = "../../data/coco/",
                 download_dataset: bool = False,
                 year: Literal['2014', '2017'] = '2017',
                 split: Literal['train', 'val', 'test'] = 'train') -> None:
        """
        An abstract class for COCO based datasets

        Args:
            root_dir: The path to the COCO images and annotations
            download_dataset: Specify whether to download the dataset if not present
            year: The year of the COCO dataset to use (2014, 2017)
            split: The split of the data to be used (train, val, test)
        """
        self.root_dir = Path('{root}/{year}'.format(root=root_dir, year=year))
        self.annotations_zip_path = Path('{root}/annotations_trainval{year}.zip'.format(root=self.root_dir, year=year))
        self.train_zip_path = Path('{root}/train{year}.zip'.format(root=self.root_dir, year=year))
        self.val_zip_path = Path('{root}/val{year}.zip'.format(root=self.root_dir, year=year))
        self.test_zip_path = Path('{root}/test{year}.zip'.format(root=self.root_dir, year=year))
        self.annotations_path = Path('{root}/annotations'.format(root=self.root_dir))
        self.images_path = Path('{root}/{split}{year}'.format(root=self.root_dir, split=split, year=year))
        self.instances_path = Path(
            '{root}/annotations/instances_{split}{year}.json'.format(root=self.root_dir, split=split, year=year))

        if download_dataset and (not os.path.exists(self.images_path) or not os.path.exists(self.annotations_path)):
            self._download_dataset(year, split)

    def _download_dataset(self, year: Literal['2014', '2017'], split: Literal['train', 'test', 'val']) -> None:
        """
        Downloads the dataset from COCO website.

        Args:
            year: The year of the dataset to download
            split: The split of the dataset to download
        """
        files = {
            'annotations': {
                'name': 'Train-Val {} Annotations'.format(year),
                'file_path': self.annotations_zip_path,
                'url': 'http://images.cocodataset.org/annotations/annotations_trainval{}.zip'.format(year),
                'unit': 'MB'
            },
            'train': {
                'name': 'Train {} Dataset'.format(year),
                'file_path': self.train_zip_path,
                'url': 'http://images.cocodataset.org/zips/train{}.zip'.format(year),
                'unit': 'GB'
            },
            'val': {
                'name': 'Validation {} Dataset'.format(year),
                'file_path': self.val_zip_path,
                'url': 'http://images.cocodataset.org/zips/val{}.zip'.format(year),
                'unit': 'GB'
            },
            'test': {
                'name': 'Test {} Dataset'.format(year),
                'file_path': self.test_zip_path,
                'url': 'http://images.cocodataset.org/zips/test{}.zip'.format(year),
                'unit': 'GB'
            }
        }
        if split == 'train' or split == 'val':
            _download_file(**files['annotations'])
            _extract_file(files['annotations']['file_path'], self.root_dir)
        _download_file(**files[split])
        _extract_file(files[split]['file_path'], self.root_dir)
        self.downloaded = True
