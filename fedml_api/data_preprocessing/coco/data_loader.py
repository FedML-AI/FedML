import torch.utils.data as data

from torchvision import transforms
from fedml_api.data_preprocessing.coco.transforms import Normalize, ToTensor, FixedResize
from fedml_api.data_preprocessing.coco.datasets import CocoDataset

def _data_transforms_coco():
    COCO_MEAN = (0.485, 0.456, 0.406)
    COCO_STD = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        FixedResize(513),
        Normalize(mean=COCO_MEAN, std=COCO_STD),
        ToTensor()
    ])

    return transform, transform

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_coco(datadir, train_bs, test_bs)

def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_coco(datadir, train_bs, test_bs)

def get_dataloader_coco(datadir, train_bs, test_bs):
    transform_train, transform_test = _data_transforms_coco()

    train_ds = CocoDataset(datadir, split='train', transform=transform_train, download_dataset=True)
    test_ds = CocoDataset(datadir, split='val', transform=transform_test, download_dataset=True)
    print(len(train_ds))
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl, train_ds.num_classes
