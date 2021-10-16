import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from glob import glob


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        #i=str(self.images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(masks_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError('No input file found in {}, make sure you put your images there'.format(masks_dir))
        logging.info('Creating dataset with {} examples'.format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH))
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        #mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        
        #assert len(mask_file) == 1, 'Either no mask or multiple masks found for the ID {}:{}'.format(name,mask_file)
        #assert len(img_file) == 1, 'Either no image or multiple images found for the ID {}:{}'.format(name,img_file) 

        for i in self.images_dir.glob(name):
            for k in self.masks_dir.glob(name+ self.mask_suffix + '.*'):
                mask = self.load(k) 


            for j in i.glob('*/'):

             
       
                img = self.load(j)

             
                

                assert img.size == mask.size, \
                    'Image and mask {} should be the same size, but are {} and {}'.format(name,img.size,mask.size)

                img = self.preprocess(img, self.scale, is_mask=False)
                mask = self.preprocess(mask, self.scale, is_mask=True)

                return {
                    'image': torch.as_tensor(img.copy()).float().contiguous(),
                    'mask': torch.as_tensor(mask.copy()).long().contiguous()
                }


class SkyDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        #super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
        super().__init__(images_dir, masks_dir, scale, mask_suffix='')
