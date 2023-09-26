# -*- coding: utf-8 -*-
"""
Return index of built in Pytorch datasets 
"""
import PIL
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from torchvision import datasets
import pdb
import torch
import torchvision.transforms as T

from PIL import Image
import medmnist
from medmnist.dataset import PneumoniaMNIST
from medmnist.evaluator import getAUC, getACC
from medmnist.info import INFO
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

          
class MedMNIST(Dataset):

    flag_to_class = {
        "pneumoniamnist": PneumoniaMNIST,
    }

    flag = 'pneumoniamnist'

    def __init__(self,
                 root,
                 split='train',
                 transform=None,
                 target_transform=None,
                 download=False):
        ''' dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation
    
        '''

        self.info = INFO[self.flag]
        self.as_rgb = True
        self.root = root
        task = self.info['task']
        num_channels = self.info['n_channels']
        num_classes = len(self.info['label'])

        if download:
            self.download()

        if not os.path.exists(
                os.path.join(self.root, "{}.npz".format(self.flag))):
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))

        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if self.split == 'train':
            self.img = npz_file['train_images']
            self.label = npz_file['train_labels']
        elif self.split == 'val':
            self.img = npz_file['val_images']
            self.label = npz_file['val_labels']
        elif self.split == 'test':
            self.img = npz_file['test_images']
            self.label = npz_file['test_labels']

    def __getitem__(self, index):
        img, target = self.img[index], self.label[index].astype(int)
        img = np.stack([img / 255.0] * (3 if self.as_rgb else 1), axis=0)
        img = img.transpose(1,2,0)
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return self.img.shape[0]
    
    def download(self):
        try:
            from torchvision.datasets.utils import download_url
            download_url(url=self.info["url"],
                         root=self.root,
                         filename="{}.npz".format(self.flag),
                         md5=self.info["MD5"])
        except:
            raise RuntimeError('Something went wrong when downloading! ' +
                               'Go to the homepage to download manually. ' +
                               'https://github.com/MedMNIST/MedMNIST')
        
class PneumoniaMNIST(MedMNIST):
    flag = "pneumoniamnist"

