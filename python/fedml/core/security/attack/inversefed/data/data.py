"""This is data.py from pytorch-examples.

Refer to
https://github.com/pytorch/examples/blob/master/super_resolution/data.py.
"""

from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, RandomCrop


from .datasets import DatasetFromFolder

def _build_bsds_sr(data_path, augmentations=True, normalize=True, upscale_factor=3, RGB=True):
    root_dir = _download_bsd300(dest=data_path)
    train_dir = join(root_dir, "train")
    crop_size = _calculate_valid_crop_size(256, upscale_factor)
    print(f'Crop size is {crop_size}. Upscaling factor is {upscale_factor} in mode {RGB}.')

    trainset = DatasetFromFolder(train_dir, replicate=200,
                                 input_transform=_input_transform(crop_size, upscale_factor),
                                 target_transform=_target_transform(crop_size), RGB=RGB)

    test_dir = join(root_dir, "test")
    validset = DatasetFromFolder(test_dir, replicate=200,
                                 input_transform=_input_transform(crop_size, upscale_factor),
                                 target_transform=_target_transform(crop_size), RGB=RGB)
    return trainset, validset

def _build_bsds_dn(data_path, augmentations=True, normalize=True, upscale_factor=1, noise_level=25 / 255, RGB=True):
    root_dir = _download_bsd300(dest=data_path)
    train_dir = join(root_dir, "train")

    crop_size = _calculate_valid_crop_size(256, upscale_factor)
    patch_size = 64
    print(f'Crop size is {crop_size} for patches of size {patch_size}. '
          f'Upscaling factor is {upscale_factor} in mode RGB={RGB}.')

    trainset = DatasetFromFolder(train_dir, replicate=200,
                                 input_transform=_input_transform(crop_size, upscale_factor, patch_size=patch_size),
                                 target_transform=_target_transform(crop_size, patch_size=patch_size),
                                 noise_level=noise_level, RGB=RGB)

    test_dir = join(root_dir, "test")
    validset = DatasetFromFolder(test_dir, replicate=200,
                                 input_transform=_input_transform(crop_size, upscale_factor),
                                 target_transform=_target_transform(crop_size),
                                 noise_level=noise_level, RGB=RGB)
    return trainset, validset


def _download_bsd300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest, exist_ok=True)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def _calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def _input_transform(crop_size, upscale_factor, patch_size=None):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        RandomCrop(patch_size if patch_size is not None else crop_size // upscale_factor),
        ToTensor(),
    ])


def _target_transform(crop_size, patch_size=None):
    return Compose([
        CenterCrop(crop_size),
        RandomCrop(patch_size if patch_size is not None else crop_size),
        ToTensor(),
    ])
