import os
import sys
import numpy as np
import logging

import torch

from pathlib import Path, PurePath
from typing import Callable, List, Any, Tuple, TypedDict, Optional, Literal

import pycocotools.mask as coco_mask
from PIL.Image import Image
from pycocotools.coco import COCO

from ..coco_base import COCOBase


class Datapoint(TypedDict):
    image: torch.FloatTensor
    mask: torch.IntTensor


class CocoSegmentation(COCOBase):
    coco: COCO
    transform: Callable
    data_idxs: List[int]
    ids_file: PurePath
    cat_ids: List[int]
    num_classes: int
    img_ids: List[int]
    target: np.ndarray

    def __init__(self,
                 root: str = '../../../data/coco/',
                 transform: Optional[Callable] = None,
                 download_dataset: bool = False,
                 year: Literal['2014', '2017'] = '2017',
                 split: Literal['train', 'test', 'val'] = 'train',
                 categories: Optional[List[str]] = None,
                 data_idxs: Optional[List[int]] = None) -> None:
        """
        The dataset class for COCO segmentation.

        Args:
            root: The path to COCO dataset
            transform: The transformations to be applied to the data points
            download_dataset: Specifies whether to download the dataset if not present.
            year: The year of the dataset to use (2014, 2017).
            split: The split of the dataset to initialize (train, test, val).
            categories: The categories of the COCO dataset.
            data_idxs: The indexes used for partitioning dataset.
        """
        super(COCOBase, self).__init__(root, download_dataset, year, split)

        if categories is None:
            categories = ['__background__', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                          'car', 'cat',
                          'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person',
                          'potted plant',
                          'sheep', 'sofa', 'tv', 'train']
        self.coco = COCO(self.instances_path)
        self.transform = transform
        self.data_idxs = data_idxs
        self.ids_file = Path('{}/{}_{}.ids'.format(self.root_dir, split, year))

        if self.downloaded:
            os.remove(self.ids_file)

        self.cat_ids = self.coco.getCatIds(catNms=categories)
        self.num_classes = len(self.cat_ids)

        if os.path.exists(self.ids_file):
            self.img_ids = torch.load(self.ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.img_ids = self.__preprocess(ids)

        if data_idxs is not None:
            self.img_ids = [self.img_ids[i] for i in data_idxs]

        self.__generate_target()

    def __preprocess(self, ids: List[int]) -> List[int]:
        """
        Pre-process the downloaded files to get the valid images that contain the specified categories

        Args:
            ids: The image ids to be processed

        Returns:
            The valid set of image ids
        """
        logging.info("Pre-processing mask, this will take a while. It only runs once for each split.")
        new_ids = []
        for i in range(len(ids)):
            img_id = ids[i]
            _, mask = self.__get_datapoint(img_id)
            if (mask > 0).sum() > 1000:
                new_ids.append(ids[i])
            done = int(50 * i / len(ids))
            sys.stdout.write(
                '\r[{}{}] {}% ({}/{})'.format('#' * done, '.' * (50 - done), int((i / len(ids)) * 100), i, len(ids)))
        sys.stdout.write('\n')
        sys.stdout.flush()
        logging.info('Found number of qualified images: {}'.format(len(new_ids)))
        torch.save(new_ids, self.ids_file)
        return new_ids

    def __get_mask(self, annotations: List[Any], height: int, width: int) -> np.ndarray:
        """
        Generates the mask from the annotations

        Args:
            annotations: Contains the segmentation mask paths.
            height: Height of the image.
            width: Width of the image.

        Returns:
            The generated mask from the annotations.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in annotations:
            rle = coco_mask.frPyObjects(ann['segmentation'], height, width)
            m = coco_mask.decode(rle)
            cat = ann['category_id']
            if cat in self.cat_ids:
                c = self.cat_ids.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def __get_datapoint(self, img_id: int) -> Tuple[Image, np.ndarray]:
        """
        Fetches the datapoint corresponding to the image id.

        Args:
            img_id: Id of the image.

        Returns:
            A tuple of image and mask.
        """
        annotations_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)

        img_metadata = self.coco.loadImgs(ids=img_id)[0]
        img_file = img_metadata['file_name']
        img = Image.open(self.images_path.joinpath(img_file)).convert('RGB')

        annotations = self.coco.loadAnns(ids=annotations_ids)
        mask = self.__get_mask(annotations, img_metadata['height'], img_metadata['width'])

        return img, mask

    def __generate_target(self) -> None:
        """
        Generates the targets used to partition data.
        """
        target = list()
        for img_id in self.img_ids:
            annotation_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)
            annotations = self.coco.loadAnns(ids=annotation_ids)
            category_list = np.asarray([ann['category_id'] for ann in annotations])
            target.append(category_list)
        self.target = np.asarray(target)

    def __getitem__(self, index: int) -> Datapoint:
        img, mask = self.__get_datapoint(self.img_ids[index])
        datapoint = {'image': img, 'label': Image.fromarray(mask)}
        if self.transform is None:
            return datapoint
        return self.transform(datapoint)

    def __len__(self) -> int:
        return len(self.img_ids)


if __name__ == '__main__':
    root_dir = './data/coco'
    train_data = CocoSegmentation(root=root_dir)
    val_Data = CocoSegmentation(root=root_dir, split='val')
