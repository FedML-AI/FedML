import torch
import torchvision.transforms as transforms

'''
preprocess reference : https://github.com/google-research/federated/blob/master/utils/datasets/cifar100_dataset.py
'''

def cifar100_transform(img_mean, img_std, train = True, crop_size = (24,24)):
    """cropping, flipping, and normalizing."""
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])


def preprocess_cifar_img(img, train):
    # scale img to range [0,1] to fit ToTensor api
    img = torch.div(img, 255.0)
    transoformed_img = torch.stack([cifar100_transform
        (i.type(torch.DoubleTensor).mean(),
            i.type(torch.DoubleTensor).std(),
            train)
        (i.permute(2,0,1)) 
        for i in img])
    return transoformed_img