import logging
import os

import torch.utils.data as data
from PIL import Image


class Landmarks(data.Dataset):
    """
    Custom dataset class for the Landmarks dataset.

    Args:
        data_dir (str): The directory containing the data files.
        allfiles (list): A list of data entries in the form of dictionaries with 'user_id', 'image_id', and 'class'.
        dataidxs (list, optional): List of data indices to select specific data entries. Defaults to None.
        train (bool, optional): Indicates whether the dataset is for training. Defaults to True.
        transform (callable, optional): A function/transform to apply to the data. Defaults to None.
        target_transform (callable, optional): A function/transform to apply to the target. Defaults to None.
        download (bool, optional): Whether to download the data. Defaults to False.
    """
    def __init__(
        self,
        data_dir,
        allfiles,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        """
        allfiles is [{'user_id': xxx, 'image_id': xxx, 'class': xxx} ...
                     {'user_id': xxx, 'image_id': xxx, 'class': xxx} ... ]
        Initialize the Landmarks dataset.

        Args:
            data_dir (str): The directory containing the data files.
            allfiles (list): A list of data entries in the form of dictionaries with 'user_id', 'image_id', and 'class'.
            dataidxs (list, optional): List of data indices to select specific data entries. Defaults to None.
            train (bool, optional): Indicates whether the dataset is for training. Defaults to True.
            transform (callable, optional): A function/transform to apply to the data. Defaults to None.
            target_transform (callable, optional): A function/transform to apply to the target. Defaults to None.
            download (bool, optional): Whether to download the data. Defaults to False.
        """
        self.allfiles = allfiles
        if dataidxs == None:
            self.local_files = self.allfiles
        else:
            self.local_files = self.allfiles[dataidxs[0] : dataidxs[1]]
            # print("self.local_files: %d, dataidxs: (%d, %d)" % (len(self.local_files), dataidxs[0], dataidxs[1]))
        self.data_dir = data_dir
        self.dataidxs = dataidxs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Get the number of data entries in the dataset.

        Returns:
            int: The number of data entries.
        """

        # if self.user_id != None:
        #     return sum([len(local_data) for local_data in self.mapping_per_user.values()])
        # else:
        #     return len(self.mapping_per_user)
        return len(self.local_files)

    def __getitem__(self, idx):
        """
        Get a data sample and its corresponding label by index.

        Args:
            idx (int): Index of the data sample to retrieve.

        Returns:
            tuple: A tuple containing the data sample and its corresponding label.
        """
        # if self.user_id != None:
        #     img_name = self.mapping_per_user[self.user_id][idx]['image_id']
        #     label = self.mapping_per_user[self.user_id][idx]['class']
        # else:
        #     img_name = self.mapping_per_user[idx]['image_id']
        #     label = self.mapping_per_user[idx]['class']
        img_name = self.local_files[idx]["image_id"]
        label = int(self.local_files[idx]["class"])

        img_name = os.path.join(self.data_dir, str(img_name) + ".jpg")

        # convert jpg to PIL (jpg -> Tensor -> PIL)
        image = Image.open(img_name)
        # jpg_to_tensor = transforms.ToTensor()
        # tensor_to_pil = transforms.ToPILImage()
        # image = tensor_to_pil(jpg_to_tensor(image))
        # image = jpg_to_tensor(image)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
