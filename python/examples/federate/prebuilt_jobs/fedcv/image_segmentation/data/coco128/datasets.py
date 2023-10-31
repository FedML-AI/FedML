# Code reference: https://github.com/deepchecks/deepchecks/blob/daedbaba3ba0e020e96de2ac0eb6a6f24d5359c5/docs/source/user-guide/vision/tutorials/plot_custom_task_tutorial.py

import contextlib
import os
import typing as t
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.utils import draw_segmentation_masks


class CocoSegmentDataset(VisionDataset):
    """An instance of PyTorch VisionData the represents the COCO128-segments dataset.
    Parameters
    ----------
    root : str
        Path to the root directory of the dataset.
    name : str
        Name of the dataset.
    train : bool
        if `True` train dataset, otherwise test dataset
    transforms : Callable, optional
        A function/transform that takes in an PIL image and returns a transformed version.
        E.g, transforms.RandomCrop
    """

    TRAIN_FRACTION = 1

    def __init__(
        self,
        root: str,
        name: str = "train2017",
        train: bool = True,
        transform: t.Optional[t.Callable] = None,
        data_idxs=None,
    ) -> None:
        super().__init__(root, transforms=transform)

        self.train = train
        self.root = Path(root).absolute()
        self.images_dir = Path(root) / "images" / name
        self.labels_dir = Path(root) / "labels" / name
        self.data_idxs = data_idxs
        self.num_classes = 80 if name == "train2017" else 20

        images: t.List[Path] = sorted(self.images_dir.glob("./*.jpg"))
        labels: t.List[t.Optional[Path]] = []

        for image in images:
            label = self.labels_dir / f"{image.stem}.txt"
            labels.append(label if label.exists() else None)

        if self.data_idxs is not None:
            images = [images[i] for i in self.data_idxs]
            labels = [labels[i] for i in self.data_idxs]

        assert len(images) != 0, "Did not find folder with images or it was empty"
        assert not all(
            l is None for l in labels
        ), "Did not find folder with labels or it was empty"

        train_len = int(self.TRAIN_FRACTION * len(images))

        if self.train is True:
            self.images = images[0:train_len]
            self.labels = labels[0:train_len]
        else:
            self.images = images[train_len:]
            self.labels = labels[train_len:]

    def __getitem__(self, idx: int) -> t.Tuple[Image.Image, np.ndarray]:
        """Get the image and label at the given index."""
        image = Image.open(str(self.images[idx]))
        # to RGB
        image = image.convert("RGB")
        label_file = self.labels[idx]

        masks = []
        classes = []
        if label_file is not None:
            for label_str in label_file.open("r").read().strip().splitlines():
                label = np.array(label_str.split(), dtype=np.float32)
                class_id = int(label[0])
                # Transform normalized coordinates to un-normalized
                coordinates = (
                    (label[1:].reshape(-1, 2) * np.array([image.width, image.height]))
                    .reshape(-1)
                    .tolist()
                )
                # Create mask image
                mask = Image.new("L", (image.width, image.height), 0)
                ImageDraw.Draw(mask).polygon(coordinates, outline=1, fill=1)
                # Add to list
                masks.append(np.array(mask, dtype=bool))
                classes.append(class_id)

        if self.transforms is not None:
            transformed = self.transforms(
                {
                    "image": image,
                    "label": masks,
                    "class_id": classes,
                    "class_num": self.num_classes,
                }
            )
            image = transformed["image"]
            masks = transformed["label"]

        return {"image": image, "label": masks, "class": classes}

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    root_dir = "/home/beiyu/fedcv_data/coco128"
    train_data = CocoSegmentDataset(root=root_dir, train=True)
    print(len(train_data))
    print(train_data.__getitem__(0))
