import glob
import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from pathlib import Path


class CityscapesSegmentation(Dataset):

    def __init__(self,
                 root_dir='../../data/cityscapes',
                 split='train_extra',
                 annotation_type='gtCoarse',
                 transform=None,
                 data_idxs=None):
        """
            The dataset class for Pascal VOC Augmented Dataset.

            Args:
                root_dir: The path to the dataset.
                split: The type of dataset to use (train, train_extra, val).
                transform: The custom transformations to be applied to the dataset.
                data_idxs: The list of indexes used to partition the dataset.
            """
        if annotation_type == 'gtFine' and split == 'train_extra':
            raise RuntimeError("Cannot have annotation type as gtFine for train_extra as the split")
        self.root_dir = root_dir
        self.annotation_type = annotation_type
        self.train_images = Path('{}/leftImg8bit/{}'.format(root_dir, split))
        self.masks_dir = Path('{}/{}/{}'.format(root_dir, annotation_type, split))
        self.targets_path = Path('{}/targets_{}_{}.pickle'.format(root_dir, split, annotation_type))
        self.id_to_train_id = {-1: 255, 0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
            7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4,
            14: 255, 15: 255, 16: 255, 17: 5,
            18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
            28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18}
        self.classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.transform = transform
        self.images = list()
        self.masks = list()
        self.targets = None

        self.__preprocess()
        if data_idxs is not None:
            self.images = [self.images[i] for i in data_idxs]
            self.masks = [self.masks[i] for i in data_idxs]

        self.__generate_targets()

    def __preprocess(self):
        for city in os.listdir(self.train_images):
            self.images.extend(sorted(glob.glob('{}/{}/*.png'.format(self.train_images, city))))
            self.masks.extend(sorted(glob.glob('{}/{}/*_{}_labelIds.png'.format(self.masks_dir, city, self.annotation_type))))
        assert len(self.images) == len(self.masks)

    def __generate_targets(self):
        targets = list()
        targets_dict = dict()
        with open(self.targets_path, 'rb') as pickle_file:
            targets_dict = pickle.load(pickle_file)
        for mask_path in self.masks:
            targets.append(targets_dict[mask_path.split('/')[-1]])
        self.targets = np.asarray(targets)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])
        sample = {'image': img, 'label': mask}

        if self.transform is not None:
            sample = self.transform(sample)
        mask = sample['label']
        for k, v in self.id_to_train_id.items():
            mask[mask == k] = v
        sample['label'] = mask
        return sample

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = CityscapesSegmentation()
    print('Train Extra/Coarse: {}'.format(len(dataset)))
    assert len(dataset) == 19998

    dataset = CityscapesSegmentation(split='train')
    print('Train/Coarse: {}'.format(len(dataset)))
    assert len(dataset) == 2975

    dataset = CityscapesSegmentation(split='val')
    print('Val/Coarse: {}'.format(len(dataset)))
    assert len(dataset) == 500

    dataset = CityscapesSegmentation(split='train', annotation_type='gtFine')
    print('Train/Fine: {}'.format(len(dataset)))
    assert len(dataset) == 2975

    dataset = CityscapesSegmentation(split='val', annotation_type='gtFine')
    print('Val/Fine: {}'.format(len(dataset)))
    assert len(dataset) == 500
