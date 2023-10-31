import os
import shutil
import torch

import numpy as np
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset

from pathlib import Path, PurePath

from .utils import _download_file, _extract_file


class PascalVocAugmentedSegmentation(Dataset):

    def __init__(self,
                 root_dir='../../data/pascal_voc_augmented',
                 split='train',
                 download_dataset=False,
                 transform=None,
                 data_idxs=None):
        """
        The dataset class for Pascal VOC Augmented Dataset.

        Args:
            root_dir: The path to the dataset.
            split: The type of dataset to use (train, test, val).
            download_dataset: Specify whether to download the dataset if not present.
            transform: The custom transformations to be applied to the dataset.
            data_idxs: The list of indexes used to partition the dataset.
        """
        self.root_dir = root_dir
        self.images_dir = Path('{}/dataset/img'.format(root_dir))
        self.masks_dir = Path('{}/dataset/cls'.format(root_dir))
        self.split_file = Path('{}/dataset/{}.txt'.format(root_dir, split))
        self.transform = transform
        self.images = list()
        self.masks = list()
        self.targets = None

        if download_dataset:
            self.__download_dataset()

        self.__preprocess()
        if data_idxs is not None:
            self.images = [self.images[i] for i in data_idxs]
            self.masks = [self.masks[i] for i in data_idxs]

        self.__generate_targets()

    def __download_dataset(self):
        """
        Downloads the PASCAL VOC Augmented dataset.
        """
        files = {
            'pascalvocaug': {
                'name': 'PASCAL Train and Test Augmented Dataset',
                'file_path': Path('{}/benchmark.tgz'.format(self.root_dir)),
                'url': 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark'
                       '.tgz',
                'unit': 'GB'
            }
        }

        _download_file(**files['pascalvocaug'])
        _extract_file(files['pascalvocaug']['file_path'], self.root_dir)
        shutil.move('{}/benchmark_RELEASE/dataset'.format(self.root_dir), self.root_dir)
        shutil.rmtree('{}/benchmark_RELEASE'.format(self.root_dir))

    def __preprocess(self):
        """
        Pre-process the dataset to get mask and file paths of the images.

        Raises:
            AssertionError: When length of images and masks differs.
        """
        with open(self.split_file, 'r') as file_names:
            for file_name in file_names:
                img_path = Path('{}/{}.jpg'.format(self.images_dir, file_name.strip(' \n')))
                mask_path = Path('{}/{}.mat'.format(self.masks_dir, file_name.strip(' \n')))
                assert os.path.isfile(img_path)
                assert os.path.isfile(mask_path)
                self.images.append(img_path)
                self.masks.append(mask_path)
            assert len(self.images) == len(self.masks)

    def __generate_targets(self):
        """
        Used to generate targets which in turn is used to partition data in an non-IID setting.
        """
        targets = list()
        for i in range(len(self.images)):
            mat = sio.loadmat(self.masks[i], mat_dtype=True, squeeze_me=True, struct_as_record=False)
            categories = mat['GTcls'].CategoriesPresent
            if isinstance(categories, np.ndarray):
                categories = np.asarray(list(categories))
            else:
                categories = np.asarray([categories]).astype(np.uint8)
            targets.append(categories)
        self.targets = np.asarray(targets)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mat = sio.loadmat(self.masks[index], mat_dtype=True, squeeze_me=True, struct_as_record=False)
        mask = mat['GTcls'].Segmentation
        mask = Image.fromarray(mask)
        sample = {'image': img, 'label': mask}
 
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        """
        Returns:
            The clasess present in the Pascal VOC Augmented dataset.
        """
        return ('__background__', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'television',
                'train')
