import os
import glob
import pickle

import numpy as np
from PIL import Image


classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
for split, t in [
    ('train', 'gtFine'),
    ('train', 'gtCoarse'),
    ('train_extra', 'gtCoarse'),
    ('val', 'gtFine'),
    ('val', 'gtCoarse'),
]:
    print('Loading {}....'.format(split))
    mask_paths = list()
    base_path = './{}/{}'.format(t, split)
    for city in os.listdir(base_path):
        mask_paths.extend(sorted(glob.glob('./{}/{}/*_{}_labelIds.png'.format(base_path, city, t))))
    print(len(mask_paths))
    values = dict()
    for mask_path in mask_paths:
        mask = np.asarray(Image.open(mask_path), dtype=np.int32)
        cats = set(np.unique(mask))
        targets = np.asarray(list(set(classes).intersection(cats))).astype(np.uint8)
        values[mask_path.split('/')[-1]] = targets
    with open('targets_{}_{}.pickle'.format(split, t), 'wb') as f:
        pickle.dump(values, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()