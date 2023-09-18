import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR100

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def default_loader(path):
    """
    Default image loader function using PIL to open and convert an image to RGB format.

    Args:
        path (str): The path to the image file.

    Returns:
        PIL.Image: The loaded image in RGB format.
    """
    return pil_loader(path)

def pil_loader(path):
    """
    Image loader function using PIL to open and convert an image to RGB format.

    Args:
        path (str): The path to the image file.

    Returns:
        PIL.Image: The loaded image in RGB format.
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class CIFAR100_truncated(data.Dataset):
    def __init__(
        self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False,
    ):
        """
        Custom dataset class for truncated CIFAR-100 dataset.

        Args:
            root (str): The root directory where the dataset is stored.
            dataidxs (list or None): List of data indices to include in the dataset. If None, includes all data.
            train (bool): Indicates whether the dataset is for training (True) or testing (False).
            transform (callable, optional): A function/transform to apply to the data.
            target_transform (callable, optional): A function/transform to apply to the target (class label).
            download (bool, optional): Whether to download the dataset if not found locally.
        """
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        """
        Build the truncated CIFAR-100 dataset by loading data based on data indices.

        Returns:
            tuple: A tuple containing the data and target (class labels).
        """
        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)

        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        """
        Truncate the green and blue channels of specified images in the dataset.

        Args:
            index (numpy.ndarray): An array of indices indicating which images to truncate.
        """
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
