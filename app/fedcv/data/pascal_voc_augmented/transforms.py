import random
import logging
import numpy as np
import torch
from PIL import ImageOps, ImageFilter, Image
from torchvision import transforms


class RandomMirror(object):
    """
    Randomly perform a lateral inversion on the image.
    """
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return {
            'image': img,
            'label': mask,
        }


class RandomScaleCrop(object):

    def __init__(self, base_size = 512, crop_size = 512):
        """
        Randomly scales and crops the image.

        Args:
            base_size: The base size to scale
            crop_size: The size to crop
        """
        self.base_size = base_size
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        width, height = img.size
        if height > width:
            output_width = short_size
            output_height = int(1.0 * height * output_width / width)
        else:
            output_height = short_size
            output_width = int(1.0 * width * output_height / height)
        img = img.resize((output_width, output_height), Image.BILINEAR)
        mask = mask.resize((output_width, output_height), Image.NEAREST)
        if short_size < self.crop_size:
            padding_height = self.crop_size - output_height if output_height < self.crop_size else 0
            padding_width = self.crop_size - output_width if output_width < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padding_width, padding_height), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padding_width, padding_height), fill=0)
        width, height = img.size
        x1 = random.randint(0, width - self.crop_size)
        y1 = random.randint(0, height - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return {
            'image': img,
            'label': mask,
        }


class RandomGaussianBlur(object):
    """
    Randomly apply a gaussian blur to the image.
    """
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return {
            'image': img,
            'label': mask,
        }


class FixedScaleCrop(object):
    def __init__(self, crop_size = 512):
        """
        Crop the image to the specified size
        Args:
            crop_size: The size of the cropped image.
        """
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        short_size = self.crop_size
        width, height = img.size
        if width > height:
            output_height = short_size
            output_width = int(1.0 * width * output_height / height)
        else:
            output_width = short_size
            output_height = int(1.0 * height * output_width / width)
        img = img.resize((output_width, output_height), Image.BILINEAR)
        mask = mask.resize((output_width, output_height), Image.NEAREST)
        width, height = img.size
        x1 = int(round((width - self.crop_size) / 2.))
        y1 = int(round((height - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return {
            'image': img,
            'label': mask,
        }


class Normalize(object):

    def __init__(self, mean = (0, 0, 0), std = (0, 0, 0)):
        """
         Normalizes using the mean and standard deviation
        Args:
            mean: The mean value of the distribution
            std: The standard value of the distribution
        """
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        img = sample['image']
        img = self.normalize(img)
        return {
            'image': img,
            'label': sample['label'],
        }


class ToTensor(object):

    def __init__(self):
        """
        Transforms the numpy array to torch tensor
        """
        self.to_tensor = transforms.ToTensor()

    def __call__(self, sample):
        img = torch.tensor(np.array(sample['image']).astype(np.float32).transpose((2, 0, 1)))
        mask = torch.tensor(np.array(sample['label']).astype(np.float32))
        return {
            'image': img,
            'label': mask,
        }
