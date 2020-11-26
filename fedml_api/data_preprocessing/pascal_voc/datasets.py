import os
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset


class PascalVocDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, dataidxs=None):
        self.images_dir = Path('{}/dataset/img'.format(root_dir))
        self.masks_dir = Path('{}/dataset/cls'.format(root_dir))
        self.split_file = Path('{}/dataset/{}.txt'.format(root_dir, split))
        self.transform = transform
        self.images = list()
        self.masks = list()
        self.targets = None
        self.__preprocess()
        if dataidxs is not None:
            self.images = [self.images[i] for i in dataidxs]
            self.masks = [self.masks[i] for i in dataidxs]

        self.__generate_targets()

    def __preprocess(self):
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
        self.targets = list()
        for i in range(len(self.images)):
            mat = sio.loadmat(self.masks[i], mat_dtype=True, squeeze_me=True, struct_as_record=False)
            categories = mat['GTcls'].CategoriesPresent
            if isinstance(categories, np.ndarray):
                categories = np.asarray(list(categories))
            else:
                categories = np.asarray([categories]).astype(np.uint8)
            self.targets.append(categories)
        self.targets = np.asarray(self.targets)

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
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'television',
                'train')
