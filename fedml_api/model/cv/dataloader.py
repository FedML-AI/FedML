import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange
import os
from pycocotools.coco import COCO
from pycocotools import mask
from torchvision import transforms
import deeplab_utils 
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class COCOSegmentation(Dataset):
    NUM_CLASSES = 21
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
        1, 64, 20, 63, 7, 72]

    def __init__(self,
                 base_dir='./coco_data/',
                 split='train',
                 year='2017'):
        super().__init__()
        ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format(split, year))
        ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}.pth'.format(split, year))
        self.img_dir = os.path.join(base_dir, '{}{}'.format(split, year))
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
          self.ids = torch.load(ids_file)
          if split == 'train':
              self.ids = self.ids[:10000]
          else:
              self.ids = self.ids[:3000]
        else:
          ids = list(self.coco.imgs.keys())
          
        # self.ids = ['000000059821', '000000399212', '000000172123', '000000179085', '000000311301', '000000059682', '000000086147', '000000407289', '000000410576', '000000478077', '000000269997', '000000284605', '000000145831', '000000250680', '000000171970', '000000175236', '000000081035', '000000053142', '000000257328', '000000200231', '000000125404', '000000401429', '000000235701', '000000311991', '000000235390', '000000158422', '000000228300', '000000328890', '000000001099', '000000053677', '000000126392', '000000298350', '000000472659', '000000378618', '000000395888', '000000207739', '000000404486', '000000443272', '000000382704', '000000100159', '000000195639', '000000536433', '000000347535', '000000207691', '000000239811', '000000303365', '000000326209', '000000418700', '000000524905', '000000027524']
        # self.ids = ['000000059821', '000000399212', '000000172123', '000000179085', '000000311301', '000000059682', '000000086147', '000000407289', '000000410576', '000000478077'] 
          self.ids = self._preprocess(ids, ids_file)
        # self.args = args
        self.base_size = 513
        self.crop_size = 513

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        width, height = _img.size
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        _target = Image.fromarray(self._gen_seg_mask(
            cocotarget, height, width))

        return _img, _target

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            deeplab_utils.RandomHorizontalFlip(),
            deeplab_utils.RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size),
            deeplab_utils.RandomGaussianBlur(),
            deeplab_utils.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            deeplab_utils.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            deeplab_utils.FixScaleCrop(crop_size=self.crop_size),
            deeplab_utils.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            deeplab_utils.ToTensor()])

        return composed_transforms(sample)


    def __len__(self):
        return len(self.ids)



if __name__ == "__main__":
    from dataloaders import custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    coco_val = COCOSegmentation(args, split='val', year='2017')

    dataloader = DataLoader(coco_val, batch_size=4, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='coco')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
