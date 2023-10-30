import os

import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class NIHChestXray(data.Dataset):
    label_header = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Hernia",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pleural_Thickening",
        "Pneumonia",
        "Pneumothorax",
    ]

    def __init__(self, data_dir, label_dir=None, dataidxs=None, train=True, transform=None, download=False):
        self.data_dir = data_dir
        self.label_dir = label_dir if label_dir is not None else data_dir
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.download = download
        self.num_classes = len(self.label_header)

        self.images, self.labels = self._build_datasets()

    def _build_datasets(self):
        images = []
        labels = []
        import csv

        with open(os.path.join(self.label_dir, "Data_Entry_2017.csv"), "r") as f:
            reader = csv.reader(f)
            _ = next(reader)
            ids = 0
            for row in reader:
                if self.dataidxs is not None:
                    if ids not in self.dataidxs:
                        ids += 1
                        continue
                ids += 1
                _img = row[0]
                _label = row[1]
                _label = _label.split("|")
                _label = [x for x in _label if x != "No Finding"]
                _label_onehot = np.zeros(len(self.label_header), dtype=float)
                for i in range(0, len(_label)):
                    _label_onehot[self.label_header.index(_label[i])] = 1
                images.append(_img)
                labels.append(_label_onehot)

        return images, labels

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])

        from PIL import Image

        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float)

        return img, label

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/cache")
    parser.add_argument("--label_dir", type=str, default="data/cache")
    args = parser.parse_args()

    args.data_dir = os.path.join("G:\\dataset\\NIH Chest X-ray\\ChestXray-NIHCC\\images")
    args.label_dir = os.path.join("G:\\dataset\\NIH Chest X-ray\\ChestXray-NIHCC")

    dataset = NIHChestXray(
        data_dir=args.data_dir,
        label_dir=args.label_dir,
    )
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
    print(dataset[3])
    print(dataset[4])
