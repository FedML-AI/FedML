import logging
from typing import Iterable, Dict

import numpy as np
import torch

from PIL import Image


class Normalize(object):
    mean: Iterable[float]
    std: Iterable[float]

    def __init__(
        self,
        mean: Iterable[float] = (0.0, 0.0, 0.0),
        std: Iterable[float] = (1.0, 1.0, 1.0),
    ):
        """
        Normalizes using the mean and standard deviation
        Args:
            mean: The mean value of the distribution
            std: The standard value of the distribution
        """
        self.mean = mean
        self.std = std

    def __call__(self, sample: Dict[str, Image.Image]) -> Dict[str, np.ndarray]:
        img = sample["image"]
        mask = sample["label"]
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        # logging.info(f"Image shape: {img.shape}")

        img /= 255.0
        img -= self.mean
        img /= self.std

        return {"image": img, "label": mask}


class ToTensor(object):
    """
    Transforms the numpy array to torch tensor
    """

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.FloatTensor]:
        img = sample["image"]
        mask = sample["label"]
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {"image": img, "label": mask}


class FixedResize(object):
    def __init__(self, size: int):
        """
        Resizes the image to the specified size.

        Args:
            size: The size to resize the image.
        """
        self.size = (size, size)

    def __call__(self, sample: Dict[str, Image.Image]) -> Dict[str, Image.Image]:
        img = sample["image"]
        mask = sample["label"]
        class_id = sample["class_id"]
        class_num = sample["class_num"]

        # assert img.size == mask.size

        img_size = list(img.size)[::-1]
        img = img.resize(self.size, Image.BILINEAR)

        # mask: List[List[bool, bool]], class_id: List[int], class_num: int
        # mask to multi-class mask
        mask = np.array(mask)
        mask_ = np.zeros(img_size, dtype=np.float32)

        for idx, class_idx in enumerate(class_id):
            mask_[mask[idx] == True] = class_idx

        # to Image
        # print(mask_.shape)
        mask = Image.fromarray(mask_.astype(np.uint8))
        mask = mask.resize(self.size, Image.NEAREST)

        return {"image": img, "label": mask}
