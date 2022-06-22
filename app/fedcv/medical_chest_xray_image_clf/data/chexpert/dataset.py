import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms


class CheXpert(data.Dataset):
    # download_url = "http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip"
    label_header = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]

    def __init__(
        self, data_dir, label_dir=None, dataidxs=None, train=True, transform=None, download=False, policy="zeros"
    ):
        """
        Args:
            data_dir (string): Path to the directory with the data.
            label_dir (string): Path to the directory with the labels.
            dataidxs (list): List of indices to use for the data.
            train (bool): Whether to load the train or test data.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on a sample's label.
            download (bool, optional): If true, downloads the dataset from the internet and puts it in
                ``data_dir``. If False, already downloaded files are expected to be in ``data_dir``.
            policy (string, optional): The policy for uncertain labels.
        """
        self.data_dir = data_dir
        self.label_dir = label_dir if label_dir is not None else data_dir
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.download = download
        self.policy = policy
        self.num_classes = len(self.label_header)

        if self.train:
            self.data_dir = os.path.join(self.data_dir, "train")
            self.label_path = os.path.join(self.label_dir, "train.csv")
        else:
            self.data_dir = os.path.join(self.data_dir, "valid")
            self.label_path = os.path.join(self.label_dir, "valid.csv")

        if self.download:
            self._download_data()

        self.images, self.labels = self._build_datasets()

        assert len(self.images) == len(self.labels)
        assert len(self.labels[0]) == len(self.label_header)

        if self.dataidxs is not None:
            self.images = [self.images[i] for i in self.dataidxs]
            self.labels = [self.labels[i] for i in self.dataidxs]

        assert len(self.images) == len(self.labels)

    def _download_data(self):
        # TODO: download and unzip the data
        pass

    def _build_datasets(self):
        images = []
        labels = []
        import csv

        with open(self.label_path, "r") as f:
            reader = csv.reader(f)
            # skip the header
            _ = next(reader)
            for row in reader:
                _img = row[0]
                _label = row[5:]

                _img = os.path.join(*_img.split("/")[2:])

                for i in range(len(_label)):
                    if _label[i] == "" or float(_label[i]) == -1:
                        _label[i] = 0 if self.policy == "zeros" else 1
                    else:
                        _label[i] = int(float(_label[i]))
                images.append(_img)
                labels.append(_label)

        return images, labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path, label = self.images[index], self.labels[index]

        img_path = os.path.join(self.data_dir, img_path)

        from PIL import Image

        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.float)

        return img, label

    def __len__(self):
        return len(self.images)
