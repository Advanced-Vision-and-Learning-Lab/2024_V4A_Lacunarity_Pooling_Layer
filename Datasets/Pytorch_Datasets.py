# -*- coding: utf-8 -*-
"""
Return index of built in Pytorch datasets 
"""
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
import pdb

import torch
import torchvision.transforms as T

from PIL import Image
import medmnist
from medmnist.evaluator import getAUC, getACC
from medmnist.info import INFO
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# -*- coding: utf-8 -*-
"""
Created on Mon July 01 16:01:36 2019
GTOS data loader
@author: jpeeples
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import pdb
import torch


class PlantLeaf(Dataset):

    def __init__(self, texture_dir, split = 'train', transform = None):  # numset: 0~5
        self.texture_dir = texture_dir
        self.img_transform = transform
        self.files = []  # empty list
        self.targets = [] #labels

        #pdb.set_trace()
        imgset_dir = os.path.join(self.texture_dir)

        if split == 'train':  # train
            #Get training file
            sample_dir = os.path.join(imgset_dir,'train','train')
            classes = sorted(os.listdir(sample_dir))
            self.classes = classes
            label = 0
            #Loop through data frame and get each image
            for img_folder in classes:
                #Set class label 
                #Select folder and remove class number/space in name
                temp_img_folder = os.path.join(sample_dir,img_folder)
                for image in os.listdir(temp_img_folder):
                    img_file = os.path.join(temp_img_folder,image)
                    self.files.append({  # appends the images
                            "img": img_file,
                            "label": label
                        })
                    self.targets.append(label)
                label +=1

        elif split == 'test':  # test
            sample_dir = os.path.join(imgset_dir,'test', 'test')
            classes = sorted(os.listdir(sample_dir))
            self.classes = classes
            label = 0
            #Loop through data frame and get each image
            for img_folder in classes:
                #Set class label 
                #Select folder and remove class number/space in name
                temp_img_folder = os.path.join(sample_dir,img_folder)
                for image in os.listdir(temp_img_folder):
                    img_file = os.path.join(temp_img_folder,image)
                    self.files.append({  # appends the images
                            "img": img_file,
                            "label": label
                        })
                    self.targets.append(label)
                label +=1
        
        elif split == 'val':  # test
            sample_dir = os.path.join(imgset_dir,'valid', 'valid')
            classes = sorted(os.listdir(sample_dir))
            self.classes = classes
            label = 0
            #Loop through data frame and get each image
            for img_folder in classes:
                #Set class label 
                #Select folder and remove class number/space in name
                temp_img_folder = os.path.join(sample_dir,img_folder)
                for image in os.listdir(temp_img_folder):
                    img_file = os.path.join(temp_img_folder,image)
                    self.files.append({  # appends the images
                            "img": img_file,
                            "label": label
                        })
                    self.targets.append(label)
                label +=1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        datafiles = self.files[index]

        img_file = datafiles["img"]
        img = Image.open(img_file)

        label_file = datafiles["label"]
        label = torch.as_tensor(label_file)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, label,index


class FashionMNIST_Index(Dataset):
    def __init__(self,directory,transform=None,train=True,download=True): 
        
        self.transform = transform
        self.images = datasets.FashionMNIST(directory,train=train,transform=transform,
                                       download=download)
        self.classes = self.images.classes
        self.targets = self.images.targets
        
    def __getitem__(self, index):
        data, target = self.images[index]
        return data, target, index

    def __len__(self):
        return len(self.images)
          
class MedMNIST(Dataset):

    flag = ...

    def __init__(self,
                 root,
                 split='train',
                 as_rgb = False,
                 transform=None,
                 target_transform=None,
                 download=True,
                 ):
        ''' dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation
    
        '''
        self.c=5
        self.info = INFO[self.flag]
        self.as_rgb = as_rgb
        self.root = root
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
        self.classes = list(self.info["label"].values())

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
        img = Image.fromarray(np.uint8(img))
        #img = img.convert('L')
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return self.img.shape[0]

    def __repr__(self):
        '''Adapted from torchvision.
        '''
        _repr_indent = 4
        head = "Dataset " + self.__class__.__name__

        body = ["Number of datapoints: {}".format(self.__len__())]
        body.append("Root location: {}".format(self.root))
        body.append("Split: {}".format(self.split))
        body.append("Task: {}".format(self.info["task"]))
        body.append("Number of channels: {}".format(self.info["n_channels"]))
        body.append("Meaning of labels: {}".format(self.info["label"]))
        body.append("Number of samples: {}".format(self.info["n_samples"]))
        body.append("Description: {}".format(self.info["description"]))
        body.append("License: {}".format(self.info["license"]))

        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)

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

class MedMNIST2D(MedMNIST):

    def __getitem__(self, index):
        '''
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        '''
        img, target = self.img[index], self.label[index].astype(int)
        img = Image.fromarray(img)
        if self.as_rgb:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def save(self, folder, postfix="png", write_csv=True):

        from medmnist.utils import save2d

        save2d(imgs=self.img,
               labels=self.label,
               img_folder=os.path.join(folder, self.flag),
               split=self.split,
               postfix=postfix,
               csv_path=os.path.join(folder, f"{self.flag}.csv") if write_csv else None)

    def montage(self, length=20, replace=False, save_folder=None):
        from medmnist.utils import montage2d

        n_sel = length * length
        sel = np.random.choice(self.__len__(), size=n_sel, replace=replace)

        montage_img = montage2d(imgs=self.img,
                                n_channels=self.info['n_channels'],
                                sel=sel)

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            montage_img.save(os.path.join(save_folder,
                                          f"{self.flag}_{self.split}_montage.jpg"))

        return montage_img


class PathMNIST(MedMNIST):
    flag = "pathmnist"

class BloodMNIST(MedMNIST):
    flag = "bloodmnist"

class OCTMNIST(MedMNIST):
    flag = "octmnist"


class PneumoniaMNIST(MedMNIST):
    flag = "pneumoniamnist"


class ChestMNIST(MedMNIST):
    flag = "chestmnist"


class DermaMNIST(MedMNIST):
    flag = "dermamnist"


class RetinaMNIST(MedMNIST):
    flag = "retinamnist"


class BreastMNIST(MedMNIST):
    flag = "breastmnist"


class OrganMNISTAxial(MedMNIST):
    flag = "organamnist"


class OrganMNISTCoronal(MedMNIST):
    flag = "organcmnist"


class OrganMNISTSagittal(MedMNIST):
    flag = "organsmnist"