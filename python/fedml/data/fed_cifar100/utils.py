import numpy as np
import torch
import torchvision.transforms as transforms

"""
preprocess reference : https://github.com/google-research/federated/blob/master/utils/datasets/cifar100_dataset.py
"""


# def cifar100_transform(img_mean, img_std, train=True, crop_size=(24, 24)):
def cifar100_transform(img_mean, img_std, train=True, crop_size=32):
    """cropping, flipping, and normalizing."""
    if train:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=img_mean, std=img_std),
                Cutout(16),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=img_mean, std=img_std),
            ]
        )
        # return transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.CenterCrop(crop_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=img_mean, std=img_std),
        #     ]
        # )


def preprocess_cifar_img(img, train):
    # scale img to range [0,1] to fit ToTensor api
    img = torch.div(img, 255.0)
    transoformed_img = torch.stack(
        [
            cifar100_transform(
                [0.5071, 0.4865, 0.4409],
                [0.2673, 0.2564, 0.2762],
                train,
            )(i.permute(2, 0, 1))
            for i in img
        ]
        # [
        #     cifar100_transform(
        #         i.type(torch.DoubleTensor).mean(),
        #         i.type(torch.DoubleTensor).std(),
        #         train,
        #     )(i.permute(2, 0, 1))
        #     for i in img
        # ]
    )
    return transoformed_img



class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


