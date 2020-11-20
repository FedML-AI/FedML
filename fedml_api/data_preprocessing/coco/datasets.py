import os
import sys
import torch
import torchvision
import numpy as np
import pycocotools.mask as coco_mask

from pathlib import Path
from pycocotools.coco import COCO
from PIL import Image
from fedml_api.data_preprocessing.coco.transforms import Normalize, ToTensor
from fedml_api.data_preprocessing.coco.utils import _download_file, _extract_file


class CocoDataset(torch.utils.data.Dataset):
    """
  COCO dataset with segmentation mask generator

  Args:
    root_dir (str, optional, default='coco_data'): The local path where the COCO data set is located.
    transform (callable, optional): The transformation to be performed on the data.
    download_dataset (bool, optional, default=False): If true downloads the dataset from the COCO website.
    year (bool, optional, default='2017'): Uses the COCO dataset from the specified year.
    categories (List[str], optional, default=['person']): The list of COCO categories to fetch from the dataset.
  """

    def __init__(self,
                 root_dir='./coco_data',
                 transform=None,
                 download_dataset=False,
                 year='2017',
                 split='train',
                 categories=['person', 'dog', 'cat'],
                 dataidxs=None):
        self.dataidxs = dataidxs  # Need torch.tensor to use this
        self.annotations_zip_path = Path('{}/annotations_trainval{}.zip'.format(root_dir, year))
        self.train_zip_path = Path('{}/train{}.zip'.format(root_dir, year))
        self.val_zip_path = Path('{}/val{}.zip'.format(root_dir, year))
        self.test_zip_path = Path('{}/test{}.zip'.format(root_dir, year))
        self.annotations_path = Path('{}/annotations'.format(root_dir))
        self.images_path = Path('{}/{}{}'.format(root_dir, split, year))
        self.instances_path = Path('{}/annotations/instances_{}{}.json'.format(root_dir, split, year))
        self.root_dir = Path('{}'.format(root_dir))
        self.ids_file = Path('{}/{}_{}.ids'.format(root_dir, split, year))

        if download_dataset and not os.path.exists(self.images_path):
            self.__download_dataset(year, split)
            if os.path.exists(self.ids_file):
                os.remove(self.ids_file)

        self.coco = COCO(self.instances_path)
        self.train_pairs = list()
        self.cat_ids = self.coco.getCatIds(catNms=categories)
        self.num_classes = len(self.cat_ids)
        if os.path.exists(self.ids_file):  # If the pre-processed id file exists
            self.img_ids = torch.load(self.ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.img_ids = self.__preprocess(ids)

        if dataidxs is not None:
            self.img_ids = [self.img_ids[i] for i in dataidxs]

        if transform is None:
            self.transform = torchvision.transforms.Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensor(),
            ])
        else:
            self.transform = transform

    def __download_dataset(self, year, split):
        """
    Downloads the dataset from COCO website.

    Args:
      year (bool, optional, default='2017'): Uses the COCO dataset from the specified year.
    """
        os.makedirs('coco_data', exist_ok=True)
        files = {
            'annotations': {
                'name': 'Train-Val {} Annotations'.format(year),
                'file_path': self.annotations_zip_path,
                'url': 'http://images.cocodataset.org/annotations/annotations_trainval{}.zip'.format(year),
                'unit': 'MB'
            },
            'train': {
                'name': 'Train {} Dataset'.format(year),
                'file_path': self.train_zip_path,
                'url': 'http://images.cocodataset.org/zips/train{}.zip'.format(year),
                'unit': 'GB'
            },
            'val': {
                'name': 'Validation {} Dataset'.format(year),
                'file_path': self.val_zip_path,
                'url': 'http://images.cocodataset.org/zips/val{}.zip'.format(year),
                'unit': 'GB'
            },
            'test': {
                'name': 'Test {} Dataset'.format(year),
                'file_path': self.test_zip_path,
                'url': 'http://images.cocodataset.org/zips/test{}.zip'.format(year),
                'unit': 'GB'
            }
        }
        if (split == 'train' or split == 'val') and not os.path.exists(self.annotations_path):
            _download_file(**files['annotations'])
            _extract_file(files['annotations']['file_path'], self.root_dir)
        _download_file(**files[split])
        _extract_file(files[split]['file_path'], self.root_dir)

    def __preprocess(self, ids):
        print("Pre-processing mask, this will take a while. It only runs once for each split.")
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
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, self.ids_file)
        return new_ids

    def __get_mask(self, annotations, height, width):
        """
    Generates the mask from the annotations

    Args:
      annotations (List): Contains the segmentation mask paths.
      height (int): Height of the image.
      width (int): Width of the image.
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

    def __get_datapoint(self, img_id):
        """
    Fetches the datapoint corresponding to the image id

    Args:
      img_id (int): Index to fetch.
    """
        annotations_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)

        img_metadata = self.coco.loadImgs(ids=img_id)[0]
        img_file = img_metadata['file_name']
        img = Image.open(self.images_path.joinpath(img_file)).convert('RGB')

        annotations = self.coco.loadAnns(ids=annotations_ids)
        mask = self.__get_mask(annotations, img_metadata['height'], img_metadata['width'])

        return img, mask

    def __get_categories_as_target(self):
        self.target = list()
        for id in self.img_ids:
            annotation_ids = self.coco.getAnnIds(imgIds=id, catIds=self.cat_ids)
            annotations = self.coco.loadAnns(ids=annotation_ids)
            category_list = np.asarray([ann['category_id'] for ann in annotations])
            self.target.append(category_list)
        self.target = np.asarray(self.target)

    def generate_target(self):
        # if len(self.cat_ids) == 1:
        #   self.target = np.expand_dims(np.repeat(self.cat_ids[0], len(self.img_ids)), axis=1)
        # else:
        self.__get_categories_as_target()

    def __getitem__(self, index):
        img, mask = self.__get_datapoint(self.img_ids[index])
        datapoint = {'image': img, 'label': Image.fromarray(mask)}
        return self.transform(datapoint)

    def __len__(self):
        return len(self.img_ids)
