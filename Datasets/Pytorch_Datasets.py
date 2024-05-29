# -*- coding: utf-8 -*-
"""
Return index of built in Pytorch datasets 
"""
import numpy as np
from torch.utils.data import Dataset
import torch
import ntpath
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import agml
from agml.utils.io import nested_file_list

device = "cuda" if torch.cuda.is_available() else "cpu"


class PlantVillage(Dataset):
    def __init__(self, root, split='train', transform=None, download=True):  
        self.transform = transform
        self.split = split
        self.images = agml.data.AgMLDataLoader('plant_village_classification', dataset_path="Datasets/PlantVillage", overwrite=False)
        self._root_path = self.images.dataset_root
        self._image_files = sorted(nested_file_list(self._root_path))
        self.classes = self.images.classes
        self.class_to_num = self.images.class_to_num
        self.targets = [self.class_to_num[os.path.basename(os.path.dirname(file_path))] for file_path in self._image_files]

    def __getitem__(self, index):
        image_path = self._image_files[index]
        image = Image.open(image_path).convert('RGB')        
        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self._image_files)



class DeepWeeds(Dataset):
    def __init__(self, root, split='train', transform=None, download=True):

        self.transform = transform
        self.split = split
        self.load = agml.data.AgMLDataLoader('rangeland_weeds_australia', dataset_path="Datasets/DeepWeeds", overwrite=False)
        self.images = ImageFolder(root,transform=transform)
        self.classes = self.images.classes
        self.targets = self.images.targets

    def __getitem__(self, index):
        data, target = self.images[index]
        
        return data, target

    def __len__(self):
        return len(self.images)
    


#This is for getting all images in a directory (including subdirs)
def getListOfFiles(dirName):
    # create a list of all files in a root dir
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles


class LeavesTex1200(Dataset):
    """1200tex - leaf textures
    Casanova, Dalcimar, Jarbas Joaci de Mesquita Sá Junior, and Odemir Martinez Bruno.
    "Plant leaf identification using Gabor wavelets."
    International Journal of Imaging Systems and Technology (2009) 
    http://scg-turing.ifsc.usp.br/data/bases/LeavesTex1200.zip
    """
    def __init__(self, root, transform=None, load_all=True, grayscale=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.grayscale=grayscale
        self._image_files = getListOfFiles(root)
        self.transform = transform
        self.load_all=load_all
        self.data = []
        self.targets = []
        self.classes = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09',
                'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20']


        if self.load_all:
            for img_name in self._image_files:
                if self.grayscale:
                    self.data.append(Image.open(img_name).convert('L').convert('RGB'))
                else:
                    self.data.append(Image.open(img_name).convert('RGB'))  
                self.targets.append(int(ntpath.basename(img_name).split('_')[0][1:]) - 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        if self.load_all:
            image = self.data[idx]
            target = self.targets[idx]
        else:
            img_name = self._image_files[idx]
            if self.grayscale:
                image = Image.open(img_name).convert('L').convert('RGB')
            else:
                image = Image.open(img_name).convert('RGB')
            target = int(ntpath.basename(img_name).split('_')[0][1:])
        
        if self.transform:
            image = self.transform(image)

        return image, target