import numpy as np
import torch

from PIL import Image

class Normalize(object):
  """Normalize a tensor image with mean and standard deviation.
  Args:
    mean (tuple): means for each channel.
    std (tuple): standard deviations for each channel.
  """
  def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
    self.mean = mean
    self.std = std

  def __call__(self, sample):
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
  """Convert ndarrays in sample to Tensors."""

  def __call__(self, sample):
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
  def __init__(self, size):
    self.size = (size, size)

  def __call__(self, sample):
    img = sample['image']
    mask = sample['label']

    assert img.size == mask.size

    img = img.resize(self.size, Image.BILINEAR)
    mask = mask.resize(self.size, Image.NEAREST)

    return {
      'image': img,
      'label': mask
    }