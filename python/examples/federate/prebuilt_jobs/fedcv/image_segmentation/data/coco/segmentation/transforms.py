from typing import Iterable, Dict

import numpy as np
import torch

from PIL.Image import Image


class Normalize(object):
    mean: Iterable[float]
    std: Iterable[float]

    def __init__(self, mean: Iterable[float] = (0., 0., 0.), std: Iterable[float] = (1., 1., 1.)):
        """
        Normalizes using the mean and standard deviation
        Args:
            mean: The mean value of the distribution
            std: The standard value of the distribution
        """
        self.mean = mean
        self.std = std

    def __call__(self, sample: Dict[str, Image]) -> Dict[str, np.ndarray]:
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {
            'image': img,
            'label': mask
        }


class ToTensor(object):
    """
    Transforms the numpy array to torch tensor
    """
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.FloatTensor]:
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {
            'image': img,
            'label': mask
        }


class FixedResize(object):
    def __init__(self, size: int):
        """
        Resizes the image to the specified size.

        Args:
            size: The size to resize the image.
        """
        self.size = (size, size)

    def __call__(self, sample: Dict[str, Image]) -> Dict[str, Image]:
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {
            'image': img,
            'label': mask
        }
