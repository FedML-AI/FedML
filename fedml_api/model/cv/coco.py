import os
import sys
import torch
import torchvision
import numpy as np
import pycocotools.mask as coco_mask
import matplotlib.pyplot as plt
import requests

from pathlib import Path
from zipfile import ZipFile
from pycocotools.coco import COCO
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from deeplab_utils import *



def convert_size(size_in_bytes, unit):
  """
  Converts the bytes to human readable size format.

  Args:
    size_in_bytes (int): The number of bytes to convert
    unit (str): The unit to convert to.
  """
  if unit == 'GB':
    return '{:.2f} GB'.format(size_in_bytes / (1024 * 1024 * 1024))
  elif unit == 'MB':
    return '{:.2f} MB'.format(size_in_bytes / (1024 * 1024))
  elif unit == 'KB':
    return '{:.2f} KB'.format(size_in_bytes / 1024)
  else:
    return '{:.2f} bytes'.format(size_in_bytes)


def download_file(name, url, file_path, unit):
  """
  Downloads the file to the path specified

  Args:
    name (str): The name to print in console while downloading.
    url (str): The url to download the file from.
    file_path (str): The local path where the file should be saved.
    unit (str): The unit to convert to.
  """
  with open(file_path, 'wb') as f:
    print('Downloading {}...'.format(name))
    response = requests.get(url, stream=True)
    if response.status_code != 200:
      raise Error('Encountered error while fetching. Status Code: {}, Error: {}'.format(response.status_code, response.content))
    total = response.headers.get('content-length')
    human_readable_total = convert_size(int(total), unit)

    if total is None:
      f.write(response.content)
    else:
      downloaded = 0
      total = int(total)
      for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
        downloaded += len(data)
        human_readable_downloaded = convert_size(int(downloaded), unit)
        f.write(data)
        done = int(50 * downloaded / total)
        sys.stdout.write('\r[{}{}] {}% ({}/{})'.format('#' * done, '.' * (50 - done), int((downloaded / total) * 100), human_readable_downloaded, human_readable_total))
        sys.stdout.flush()
  sys.stdout.write('\n')
  print('Download Completed.')



def extract_file(file_path, extract_dir):
  """
  Extracts the file to the specified path.

  Args:
    file_path (str): The local path where the zip file is located.
    extract_dir (str): The local path where the files must be extracted.
  """
  with ZipFile(file_path, 'r') as zip:
    print('Extracting {} to {}...'.format(file_path, extract_dir))
    zip.extractall(extract_dir)
    zip.close()
    os.remove(file_path)
    print('Extracted {}'.format(file_path))

class CocoDataset(torch.utils.data.Dataset):
  """
  COCO dataset with segmentaiton mask generator

  Args:
    root_dir (str, optional, default='coco_data'): The local path where the COCO data set is located.
    transform (callable, optional): The transformation to be performed on the data.
    download_dataset (bool, optional, default=False): If true downloads the dataset from the COCO website.
    year (bool, optional, default='2017'): Uses the COCO dataset from the specified year.
    categories (List[str], optional, default=['person']): The list of COCO categories to fetch from the dataset.
  """
  NUM_CLASSES = 21
  CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,1, 64, 20, 63, 7, 72]
  
  def __init__(self,
               root_dir='coco_data',
               transform=None,
               download_dataset=False,
               year='2017',
               split='train',
               categories=['person']):
    self.instances_path = Path('./{}/annotations/instances_train{}.json'.format(root_dir, year))
    self.annotations_zip_path = Path('./{}/annotations_trainval{}.zip'.format(root_dir, year))
    self.train_zip_path = Path('./{}/train{}.zip'.format(root_dir, year))
    self.annotations_path = Path('./{}/annotations'.format(root_dir))
    self.train_path = Path('./{}/train{}'.format(root_dir, year))
    self.root_dir = Path('./{}'.format(root_dir))

    if download_dataset:
      self.__download_dataset(year)

    self.coco = COCO(self.instances_path)
    self.train_pairs = list()
    # self.cat_ids = self.coco.getCatIds(catNms=categories)
    self.cat_ids = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,1, 64, 20, 63, 7, 72]
    self.img_ids = self.coco.getImgIds(catIds=self.cat_ids)
    self.base_size = 513
    self.crop_size = 513

    # if transform == None:
    #   self.transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #   ])
    
  def __download_dataset(self, year):
    """
    Downloads the dataset from COCO website.

    Args:
      year (bool, optional, default='2017'): Uses the COCO dataset from the specified year.
    """
    os.makedirs('coco_data', exist_ok=True)
    files = [
      {
          'name': 'Train {} Annotations'.format(year),
          'file_path': self.annotations_zip_path,
          'url': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
          'unit': 'MB'
      },
      {
          'name': 'Train {} Dataset'.format(year),
          'file_path': self.train_zip_path,
          'url': 'http://images.cocodataset.org/zips/train2017.zip',
          'unit': 'GB'
      }
    ]
    for f in files:
      download_file(**f)
      extract_file(f['file_path'], self.root_dir)

  def __get_mask(self, annotations, height, width):
    """
    Generates the mask from the annotations

    Args:
      annotations (List): Contains the segmentation mask paths.
      height (int): Height of the image.
      wisdth (int): Width of the image.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
      rle = coco_mask.frPyObjects(ann['segmentation'], height, width)
      m = coco_mask.decode(rle)
      cat = ann['category_id']
      if cat in self.cat_ids:
        c = self.cat_ids.index(cat) + 1
      else:
        continue
      if len(m.shape) < 3:
        mask[:, :] += (mask == 0) * (m * c)
      else:
        mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
    return Image.fromarray(mask)

  def __get_datapoint(self, index):
    """
    Fetches the datapoint corresponding to the index

    Args:
      index (int): Index to fetch.
    """
    img_id = self.img_ids[index]
    annotations_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)

    img_metadata = self.coco.loadImgs(ids=img_id)[0]
    img_path = img_metadata['file_name']
    img = Image.open(self.train_path.joinpath(img_path)).convert('RGB')

    annotations = self.coco.loadAnns(ids=annotations_ids)
    mask = self.__get_mask(annotations, img_metadata['height'], img_metadata['width'])

    return img, mask

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size),
            RandomGaussianBlur(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            FixScaleCrop(crop_size=self.args.crop_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)
  
  def __getitem__(self, index):
    img, mask = self.__get_datapoint(index)
    sample = {'image': img, 'label': mask}

    # return self.transform(img), torchvision.transforms.ToTensor()(mask), img
    if self.split == "train":
        return self.transform_tr(sample)
    elif self.split == 'val':
        return self.transform_val(sample)

  def __len__(self):
    return len(self.img_ids)