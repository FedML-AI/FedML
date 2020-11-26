import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class PascalVocDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, dataidxs=None):
        self.images_dir = Path('{}/JPEGImages'.format(root_dir))
        self.masks_dir = Path('{}/SegmentationClass'.format(root_dir))
        self.split_file = Path('{}/ImageSets/Segmentation/{}.txt'.format(root_dir, split))
        self.num_classes = 21
        self.transform = transform
        self.images = list()
        self.masks = list()
        self.__preprocess()
        if dataidxs is not None:
            self.images = [self.images[i] for i in dataidxs]
            self.masks = [self.masks[i] for i in dataidxs]

    def __preprocess(self):
        with open(self.split_file, 'r') as file_names:
            for file_name in file_names:
                img_path = Path('{}/{}.jpg'.format(self.images_dir, file_name.strip(' \n')))
                mask_path = Path('{}/{}.png'.format(self.masks_dir, file_name.strip(' \n')))
                assert os.path.isfile(img_path)
                assert os.path.isfile(mask_path)
                self.images.append(img_path)
                self.masks.append(mask_path)
            assert len(self.images) == len(self.masks)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])
        sample = {'image': img, 'label': mask}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv')
