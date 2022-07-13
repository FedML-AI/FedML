"""Repeatable code parts concerning data loading."""


import torch
import torchvision
import torchvision.transforms as transforms

import os

from ..consts import *

from .data import _build_bsds_sr, _build_bsds_dn
from .loss import Classification, PSNR


def construct_dataloaders(dataset, defs, data_path='~/data', shuffle=True, normalize=True):
    """Return a dataloader with given dataset and augmentation, normalize data?."""
    path = os.path.expanduser(data_path)

    if dataset == 'CIFAR10':
        trainset, validset = _build_cifar10(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'CIFAR100':
        trainset, validset = _build_cifar100(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'MNIST':
        trainset, validset = _build_mnist(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'MNIST_GRAY':
        trainset, validset = _build_mnist_gray(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'ImageNet':
        trainset, validset = _build_imagenet(path, defs.augmentations, normalize)
        loss_fn = Classification()
    elif dataset == 'BSDS-SR':
        trainset, validset = _build_bsds_sr(path, defs.augmentations, normalize, upscale_factor=3, RGB=True)
        loss_fn = PSNR()
    elif dataset == 'BSDS-DN':
        trainset, validset = _build_bsds_dn(path, defs.augmentations, normalize, noise_level=25 / 255, RGB=False)
        loss_fn = PSNR()
    elif dataset == 'BSDS-RGB':
        trainset, validset = _build_bsds_dn(path, defs.augmentations, normalize, noise_level=25 / 255, RGB=True)
        loss_fn = PSNR()

    if MULTITHREAD_DATAPROCESSING:
        num_workers = min(torch.get_num_threads(), MULTITHREAD_DATAPROCESSING) if torch.get_num_threads() > 1 else 0
    else:
        num_workers = 0

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(defs.batch_size, len(trainset)),
                                              shuffle=shuffle, drop_last=True, num_workers=num_workers, pin_memory=PIN_MEMORY)
    validloader = torch.utils.data.DataLoader(validset, batch_size=min(defs.batch_size, len(trainset)),
                                              shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)

    return loss_fn, trainloader, validloader


def _build_cifar10(data_path, augmentations=True, normalize=True):
    """Define CIFAR-10 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if cifar10_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar10_mean, cifar10_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_cifar100(data_path, augmentations=True, normalize=True):
    """Define CIFAR-100 with everything considered."""
    # Load data
    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if cifar100_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = cifar100_mean, cifar100_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_mnist(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if mnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = mnist_mean, mnist_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_mnist_gray(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    if mnist_mean is None:
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
    else:
        data_mean, data_std = mnist_mean, mnist_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _build_imagenet(data_path, augmentations=True, normalize=True):
    """Define ImageNet with everything considered."""
    # Load data
    trainset = torchvision.datasets.ImageNet(root=data_path, split='train', transform=transforms.ToTensor())
    validset = torchvision.datasets.ImageNet(root=data_path, split='val', transform=transforms.ToTensor())

    if imagenet_mean is None:
        data_mean, data_std = _get_meanstd(trainset)
    else:
        data_mean, data_std = imagenet_mean, imagenet_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset


def _get_meanstd(dataset):
    cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
    data_mean = torch.mean(cc, dim=1).tolist()
    data_std = torch.std(cc, dim=1).tolist()
    return data_mean, data_std
