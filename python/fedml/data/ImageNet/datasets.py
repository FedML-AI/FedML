import os
import os.path

import torch.utils.data as data
from PIL import Image


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    """Find class names from subdirectories in a given directory.

    Args:
        dir (str): The root directory containing subdirectories, each representing a class.

    Returns:
        list: A sorted list of class names.
        dict: A dictionary mapping class names to their respective indices.
    """
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    """Create a dataset of image file paths and their corresponding class indices.

    Args:
        dir (str): The root directory containing subdirectories, each representing a class.
        class_to_idx (dict): A dictionary mapping class names to their respective indices.
        extensions (tuple): A tuple of allowed file extensions.

    Returns:
        list: A list of tuples, each containing the file path and class index.
        dict: A dictionary mapping class indices to the number of samples per class.
        dict: A dictionary mapping class indices to data index ranges.
    """
    images = []

    data_local_num_dict = dict()
    net_dataidx_map = dict()
    sum_temp = 0
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        target_num = 0
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    target_num += 1

        net_dataidx_map[class_to_idx[target]] = (sum_temp, sum_temp + target_num)
        data_local_num_dict[class_to_idx[target]] = target_num
        sum_temp += target_num

    assert len(images) == sum_temp
    return images, data_local_num_dict, net_dataidx_map


def pil_loader(path):
    """Load an image using PIL (Python Imaging Library).

    Args:
        path (str): The path to the image file.

    Returns:
        PIL.Image.Image: The loaded image in RGB format.
    """
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """Load an image using AccImage (optimized for CUDA).

    Args:
        path (str): The path to the image file.

    Returns:
        accimage.Image: The loaded image using AccImage.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        pass


def default_loader(path):
    """Load an image using the default loader (PIL or AccImage).

    Args:
        path (str): The path to the image file.

    Returns:
        PIL.Image.Image or accimage.Image: The loaded image.
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageNet(data.Dataset):
    def __init__(
        self,
        data_dir,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        """
        Initialize the ImageNet dataset.

        Args:
            data_dir (str): Root directory of the dataset.
            dataidxs (int or list, optional): List of indices to select specific data subsets.
            train (bool, optional): If True, loads the training dataset; otherwise, loads the validation dataset.
            transform (callable, optional): A function/transform to apply to the data.
            target_transform (callable, optional): A function/transform to apply to the labels.
            download (bool, optional): Whether to download the dataset if it's not found locally.

        Note:
            Generating this class too many times will be time-consuming.
            It's better to call this once and use ImageNet_truncated.

        """
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.loader = default_loader
        if self.train:
            self.data_dir = os.path.join(data_dir, "train")
        else:
            self.data_dir = os.path.join(data_dir, "val")

        (
            self.all_data,
            self.data_local_num_dict,
            self.net_dataidx_map,
        ) = self.__getdatasets__()

        if dataidxs is None:
            self.local_data = self.all_data
        elif isinstance(dataidxs, int):
            (begin, end) = self.net_dataidx_map[dataidxs]
            self.local_data = self.all_data[begin:end]
        else:
            self.local_data = []
            for idxs in dataidxs:
                (begin, end) = self.net_dataidx_map[idxs]
                self.local_data += self.all_data[begin:end]

    def get_local_data(self):
        return self.local_data

    def get_net_dataidx_map(self):
        return self.net_dataidx_map

    def get_data_local_num_dict(self):
        return self.data_local_num_dict


    def __getdatasets__(self):

        classes, class_to_idx = find_classes(self.data_dir)

        all_data, data_local_num_dict, net_dataidx_map = make_dataset(
            self.data_dir, class_to_idx, IMG_EXTENSIONS
        )
        if len(all_data) == 0:
            raise RuntimeError(
                f"Found 0 files in subfolders of: {self.data_dir}\n"
                f"Supported extensions are: {','.join(IMG_EXTENSIONS)}"
            )
        return all_data, data_local_num_dict, net_dataidx_map

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the index of the target class.
        """

        path, target = self.local_data[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.local_data)


class ImageNet_truncated(data.Dataset):
    def __init__(
        self,
        imagenet_dataset,
        dataidxs,
        net_dataidx_map,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        """
        Initialize a truncated version of the ImageNet dataset.

        Args:
            imagenet_dataset (ImageNet): The original ImageNet dataset.
            dataidxs (int or list): List of indices to select specific data subsets.
            net_dataidx_map (dict): Mapping of data indices in the original dataset.
            train (bool, optional): If True, loads the training dataset; otherwise, loads the validation dataset.
            transform (callable, optional): A function/transform to apply to the data.
            target_transform (callable, optional): A function/transform to apply to the labels.
            download (bool, optional): Whether to download the dataset if it's not found locally.

        """
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.net_dataidx_map = net_dataidx_map
        self.loader = default_loader
        self.all_data = imagenet_dataset.get_local_data()

        if dataidxs is None:
            self.local_data = self.all_data
        elif isinstance(dataidxs, int):
            (begin, end) = self.net_dataidx_map[dataidxs]
            self.local_data = self.all_data[begin:end]
        else:
            self.local_data = []
            for idxs in dataidxs:
                (begin, end) = self.net_dataidx_map[idxs]
                self.local_data += self.all_data[begin:end]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the index of the target class.
        """
        
        path, target = self.local_data[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.local_data)
    