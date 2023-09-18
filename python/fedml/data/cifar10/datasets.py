import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR10


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
    Default loader function for loading images.

    Args:
        path (str): Path to the image file.

    Returns:
        PIL.Image.Image: Loaded image in RGB format.
    """
    return pil_loader(path)

def pil_loader(path):
    """
    Custom PIL image loader function.

    Args:
        path (str): Path to the image file.

    Returns:
        PIL.Image.Image: Loaded image in RGB format.
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class CIFAR10_truncated(data.Dataset):
    """
    Custom dataset class for truncated CIFAR-10 data.

    Args:
        root (str): Root directory where CIFAR-10 dataset is located.
        dataidxs (list, optional): List of data indices to include (default: None).
        train (bool, optional): Whether the dataset is for training (default: True).
        transform (callable, optional): Optional transform to be applied to the image (default: None).
        target_transform (callable, optional): Optional transform to be applied to the target (default: None).
        download (bool, optional): Whether to download the dataset if not found (default: False).
    """

    def __init__(
        self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False,
    ):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        """
        Build the truncated CIFAR-10 dataset.

        Returns:
            tuple: Tuple containing data and target arrays.
        """
        print("download = " + str(self.download))
        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        """
        Truncate channels (G and B) in the images specified by the given index.

        Args:
            index (numpy.ndarray): Array of indices specifying which images to truncate.
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
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)
    