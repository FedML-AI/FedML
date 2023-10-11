import copy
import pickle

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torchvision.datasets import MNIST, EMNIST, CIFAR10

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
    return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class MNIST_truncated(data.Dataset):
    def __init__(
        self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False,
    ):
        """
        Initialize the MNIST_truncated dataset.

        Args:
            root (str): Root directory where the dataset is stored.
            dataidxs (list, optional): List of data indices to include in the dataset.
            train (bool, optional): Whether to load the training or testing data.
            transform (callable, optional): A function/transform to apply to the data.
            target_transform (callable, optional): A function/transform to apply to the target.
            download (bool, optional): Whether to download the dataset if it's not found.
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
        Build the truncated dataset based on the provided data indices.

        Returns:
            torch.Tensor: The truncated data.
            torch.Tensor: The corresponding labels/targets.
        """

        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            data = mnist_dataobj.train_data
            target = mnist_dataobj.train_labels
        else:
            data = mnist_dataobj.test_data
            target = mnist_dataobj.test_labels

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class EMNIST_truncated(data.Dataset):
    def __init__(
        self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False,
    ):
        """
        Initialize the EMNIST_truncated dataset.

        Args:
            root (str): Root directory where the dataset is stored.
            dataidxs (list, optional): List of data indices to include in the dataset.
            train (bool, optional): Whether to load the training or testing data.
            transform (callable, optional): A function/transform to apply to the data.
            target_transform (callable, optional): A function/transform to apply to the target.
            download (bool, optional): Whether to download the dataset if it's not found.
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
        Build the truncated dataset based on the provided data indices.

        Returns:
            torch.Tensor: The truncated data.
            torch.Tensor: The corresponding labels/targets.
        """
        emnist_dataobj = EMNIST(
            self.root,
            split="digits",
            train=self.train,
            transform=self.transform,
            target_transform=self.target_transform,
            download=self.download,
        )

        if self.train:
            data = emnist_dataobj.train_data
            target = emnist_dataobj.train_labels
        else:
            data = emnist_dataobj.test_data
            target = emnist_dataobj.test_labels

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def get_ardis_dataset():
    """Load the ARDIS dataset and prepare it for training.

    This function loads the ARDIS dataset from CSV files, reshapes the images,
    and prepares the dataset for training.

    Returns:
        torch.utils.data.Dataset: The ARDIS dataset prepared for training.
    """
    # Load the data from CSV files
    ardis_images = np.loadtxt("./../../../data/edge_case_examples/ARDIS/ARDIS_train_2828.csv", dtype="float")
    ardis_labels = np.loadtxt("./../../../data/edge_case_examples/ARDIS/ARDIS_train_labels.csv", dtype="float")

    # Reshape the images to [samples][width][height]
    ardis_images = ardis_images.reshape(ardis_images.shape[0], 28, 28).astype("float32")

    # Labels are one-hot encoded; extract images and labels for digit 7
    indices_seven = np.where(ardis_labels[:, 7] == 1)[0]
    images_seven = ardis_images[indices_seven, :]
    images_seven = torch.tensor(images_seven).type(torch.uint8)
    labels_seven = torch.tensor([7 for _ in ardis_labels])

    # Create an EMNIST dataset for digit 7
    ardis_dataset = EMNIST(
        "./../../../data",
        split="digits",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )

    # Set the data and targets to the extracted images and labels
    ardis_dataset.data = images_seven
    ardis_dataset.targets = labels_seven

    return ardis_dataset

def get_southwest_dataset(attack_case="normal-case"):
    """Load the Southwest dataset for a specified attack case.

    This function loads the Southwest dataset for a given attack case.

    Args:
        attack_case (str): The attack case to load. Options are "normal-case" and "almost-edge-case".

    Returns:
        pickle.Unpickler: The loaded Southwest dataset for the specified attack case.
    """
    if attack_case == "normal-case":
        with open(
            "./../../../data/edge_case_examples/southwest_cifar10/southwest_images_honest_full_normal.pkl", "rb",
        ) as train_f:
            saved_southwest_dataset_train = pickle.load(train_f)
    elif attack_case == "almost-edge-case":
        with open(
            "./../../../data/edge_case_examples/southwest_cifar10/southwest_images_honest_almost_edge_case.pkl", "rb",
        ) as train_f:
            saved_southwest_dataset_train = pickle.load(train_f)
    else:
        saved_southwest_dataset_train = None
    return saved_southwest_dataset_train


class EMNIST_NormalCase_truncated(data.Dataset):
    """
    Dataset class for normal case attack where normal
    users also hold the poisoned data point with true label.
    """

    def __init__(
        self,
        root,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        user_id=0,
        num_total_users=3383,
        poison_type="ardis",
        ardis_dataset_train=None,
        attack_case="normal-case",
    ):
        """
        Initializes the EMNIST_NormalCase_truncated dataset.

        Args:
            root (str): Root directory where the dataset is stored.
            dataidxs (list, optional): List of indices to select specific data points. Default is None.
            train (bool): True for training dataset, False for testing dataset.
            transform (callable, optional): A function/transform to apply to the data. Default is None.
            target_transform (callable, optional): A function/transform to apply to the target. Default is None.
            download (bool): Whether to download the dataset if it's not found in the root directory. Default is False.
            user_id (int): ID of the user accessing the dataset.
            num_total_users (int): Total number of users in the scenario.
            poison_type (str): Type of poisoning data. Default is "ardis".
            ardis_dataset_train (torch.utils.data.Dataset): ARDIS dataset used for poisoning. Default is None.
            attack_case (str): The type of attack case. Options are "normal-case" and "almost-edge-case". Default is "normal-case".
        """
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if attack_case == "normal-case":
            self._num_users_hold_edge_data = int(
                3383 / 20
            )  # We allow 1/20 of the users (other than the attacker) to hold the edge data.
        else:
            # Almost edge case
            self._num_users_hold_edge_data = 66  # ~2% of users hold data

        if poison_type == "ardis":
            self.ardis_dataset_train = ardis_dataset_train
            partition = np.array_split(
                np.arange(self.ardis_dataset_train.data.shape[0]), int(self._num_users_hold_edge_data),
            )

            if user_id in np.arange(self._num_users_hold_edge_data):
                user_partition = partition[user_id]
                self.saved_ardis_dataset_train = self.ardis_dataset_train.data[user_partition]
                self.saved_ardis_label_train = self.ardis_dataset_train.targets[user_partition]
            else:
                user_partition = []
                self.saved_ardis_dataset_train = self.ardis_dataset_train.data[user_partition]
                self.saved_ardis_label_train = self.ardis_dataset_train.targets[user_partition]
        else:
            raise NotImplementedError("Unsupported poison type for normal case attack ...")

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        """
        Builds the truncated dataset by combining the EMNIST dataset with the ARDIS dataset.

        Returns:
            np.ndarray: Combined data.
            np.ndarray: Combined target labels.
        """
        emnist_dataobj = EMNIST(
            self.root,
            split="digits",
            train=self.train,
            transform=self.transform,
            target_transform=self.target_transform,
            download=self.download,
        )

        if self.train:
            data = emnist_dataobj.data
            target = np.array(emnist_dataobj.targets)
        else:
            data = emnist_dataobj.data
            target = np.array(emnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        data = np.append(data, self.saved_ardis_dataset_train, axis=0)
        target = np.append(target, self.saved_ardis_label_train, axis=0)
        return data, target

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


class CIFAR10_truncated(data.Dataset):
    """
    Dataset class for a truncated version of the CIFAR-10 dataset.

    This class allows you to create a truncated version of the CIFAR-10 dataset
    by selecting specific data indices.

    Args:
        root (str): Root directory where the dataset is stored.
        dataidxs (list, optional): List of indices to select specific data points. Default is None.
        train (bool): True for training dataset, False for testing dataset.
        transform (callable, optional): A function/transform to apply to the data. Default is None.
        target_transform (callable, optional): A function/transform to apply to the target. Default is None.
        download (bool): Whether to download the dataset if it's not found in the root directory. Default is False.
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
        Builds the truncated dataset by selecting specific data indices.

        Returns:
            np.ndarray: Combined data.
            np.ndarray: Combined target labels.
        """
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


class CIFAR10NormalCase_truncated(data.Dataset):
    """
    Dataset class for normal case attack where normal
    users also hold the poisoned data point with true label.
    """

    def __init__(
        self,
        root,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        user_id=0,
        num_total_users=200,
        poison_type="southwest",
        ardis_dataset_train=None,
        attack_case="normal-case",
    ):
        """
        Initializes the CIFAR10NormalCase_truncated dataset.

        Args:
            root (str): Root directory where the dataset is stored.
            dataidxs (list, optional): List of indices to select specific data points. Default is None.
            train (bool): True for training dataset, False for testing dataset.
            transform (callable, optional): A function/transform to apply to the data. Default is None.
            target_transform (callable, optional): A function/transform to apply to the target. Default is None.
            download (bool): Whether to download the dataset if it's not found in the root directory. Default is False.
            user_id (int): ID of the user accessing the dataset.
            num_total_users (int): Total number of users in the scenario.
            poison_type (str): Type of poisoning data. Default is "southwest".
            ardis_dataset_train (np.ndarray): ARDIS dataset used for poisoning. Default is None.
            attack_case (str): The type of attack case. Options are "normal-case" and "almost-edge-case". Default is "normal-case".
        """

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self._DA_ratio = 4  # we hard code this argument for now
        if attack_case == "normal-case":
            self._num_users_hold_edge_data = (
                10  # we allow 5% of the users (other than the attacker) to hold the edge data.
            )
        elif attack_case == "almost-edge-case":
            self._num_users_hold_edge_data = (
                5  # we allow 2.5% of the users (other than the attacker) to hold the edge data.
            )
        else:
            NotImplementedError("Unsupported attacking case ...")

        self.saved_southwest_dataset_train = copy.deepcopy(ardis_dataset_train)

        if poison_type == "southwest":
            partition = np.array_split(
                np.arange(int(self.saved_southwest_dataset_train.shape[0] / self._DA_ratio)),
                int(self._num_users_hold_edge_data),
            )

            self.__aggregated_mapped_user_partition = []
            # the maped sampling thing will happen here:
            # we just generate `self.__aggregated_mapped_user_partition` once
            prev_user_counter = 0

            for bi_index, bi in enumerate(partition):
                mapped_user_partition = []
                for idx, up in enumerate(bi):
                    mapped_user_partition.extend(
                        [prev_user_counter + idx * self._DA_ratio + i for i in range(self._DA_ratio)]
                    )
                prev_user_counter += len(bi) * self._DA_ratio
                self.__aggregated_mapped_user_partition.append(mapped_user_partition)

            if user_id in np.arange(self._num_users_hold_edge_data):
                user_partition = self.__aggregated_mapped_user_partition[user_id]
                print("######### user_partition: {}, user id: {}".format(user_partition, user_id))

                self.saved_southwest_dataset_train = self.saved_southwest_dataset_train[user_partition, :, :, :]
                self.saved_southwest_label_train = 0 * np.ones(
                    (self.saved_southwest_dataset_train.shape[0],), dtype=int
                )
            else:
                user_partition = []
                self.saved_southwest_dataset_train = self.saved_southwest_dataset_train[user_partition, :, :, :]
                self.saved_southwest_label_train = 0 * np.ones(
                    (self.saved_southwest_dataset_train.shape[0],), dtype=int
                )
        else:
            NotImplementedError("Unsupported poison type for normal case attack ...")

        self.data, self.target = self.__build_truncated_dataset__()

        # if self.dataidxs is not None:
        #    print("$$$$$$$$ Inside data loader: user ID: {}, Combined data: {}, Ori data shape: {}".format(
        #                user_id, self.data.shape, len(dataidxs)))

    def __build_truncated_dataset__(self):
        """
        Builds the truncated dataset by combining the CIFAR-10 dataset with the poisoned ARDIS dataset.

        Returns:
            np.ndarray: Combined data.
            np.ndarray: Combined target labels.
        """

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

        data = np.append(data, self.saved_southwest_dataset_train, axis=0)
        target = np.append(target, self.saved_southwest_label_train, axis=0)
        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR10_Poisoned(data.Dataset):
    """
    The main motivation for this object is to adopt different transform on the mixed poisoned dataset:
    e.g. there are `M` good examples.md and `N` poisoned examples.md in the poisoned dataset.

    """

    def __init__(
        self,
        root,
        clean_indices,
        poisoned_indices,
        dataidxs=None,
        train=True,
        transform_clean=None,
        transform_poison=None,
        target_transform=None,
        download=False,
    ):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform_clean = transform_clean
        self.transform_poison = transform_poison
        self.target_transform = target_transform
        self.download = download
        self._clean_indices = clean_indices
        self._poisoned_indices = poisoned_indices

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform_clean, self.target_transform, self.download,)

        self.data = cifar_dataobj.data
        self.target = np.array(cifar_dataobj.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # we always assume that the transform function is not None
        if index in self._clean_indices:
            img = self.transform_clean(img)
        elif index in self._poisoned_indices:
            img = self.transform_poison(img)
        else:
            raise NotImplementedError("Indices should be in clean or poisoned!")

        # if index in self.transform is not None:
        #    img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)


class ImageFolderTruncated(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self, root, dataidxs=None, transform=None, target_transform=None, loader=default_loader, is_valid_file=None,
    ):
        super(ImageFolderTruncated, self).__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
        self.dataidxs = dataidxs

        ### we need to fetch training labels out here:
        self._train_labels = np.array([tup[-1] for tup in self.imgs])

        self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.dataidxs is not None:
            # self.imgs = self.imgs[self.dataidxs]
            self.imgs = [self.imgs[idx] for idx in self.dataidxs]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    @property
    def get_train_labels(self):
        return self._train_labels
