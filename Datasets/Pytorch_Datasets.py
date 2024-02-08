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
import torchgeo.datasets as geodatasets
import torch
import kornia.augmentation as K
import json
from sklearn import preprocessing
import ntpath


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


#this is for getting all images in a directory (including subdirs)
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

        return image, target, idx


class KTH_TIPS_2b_data(Dataset):

    def __init__(self, data_dir, train=True, img_transform=None, 
                 train_setting=None, test_setting=None):

        self.data_dir = data_dir
        self.img_transform = img_transform
        self.train_setting = train_setting
        self.test_setting = test_setting
        self.files = []
        self.targets = []
        self.classes = []
        #self.classes = ['aluminium_foil','brown_bread', 'corduroy', 'cork', 'cotton', 'cracker', 'lettuce_leaf', 'linen', 'white_bread', 'wood', 'wool']

        imgset_dir = os.path.join(self.data_dir, 'Images')
        # indexing variable for label
        temp_label = 0
        for file in os.listdir(imgset_dir):
            if not file.startswith('.'):
                # Set class label
                label_name = file
                self.classes.append(label_name)
                # Look inside each folder and grab samples
                texture_dir = os.path.join(imgset_dir, label_name)
                # pdb.set_trace()
                if train:
                    for ii in range(0, len(train_setting)):
                        # Only look at training samples of interest
                        sample_dir = os.path.join(texture_dir, 'sample_' + str(''.join(train_setting[ii])))
                        for image in os.listdir(sample_dir):
                            if not image.startswith('.'):
                                img_file = os.path.join(sample_dir, image)
                                label = temp_label
                                self.files.append({
                                    "img": img_file,
                                    "label": label
                                })
                                self.targets.append(label)
                else:
                    for ii in range(0, len(test_setting)):
                        # Only look at testing samples of interest
                        sample_dir = os.path.join(texture_dir, 'sample_' + str(''.join(test_setting[ii])))
                        for image in os.listdir(sample_dir):
                            if not image.startswith('.'):
                                img_file = os.path.join(sample_dir, image)
                                label = temp_label
                                self.files.append({
                                    "img": img_file,
                                    "label": label
                                })
                                self.targets.append(label)
                temp_label = temp_label + 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        datafiles = self.files[index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')

        label_file = datafiles["label"]
        label = torch.tensor(label_file)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, label, index

class GTOS_mobile_single_data(Dataset):

    def __init__(self, texture_dir, train = True,image_size = 256, img_transform = None):  # numset: 0~5
        self.texture_dir = texture_dir
        self.img_transform = img_transform
        self.files = []  # empty list
        self.targets = [] #labels

        #pdb.set_trace()
        imgset_dir = os.path.join(self.texture_dir)

        if train:  # train
            #Get training file
            sample_dir = os.path.join(imgset_dir,'train')
            class_names = sorted(os.listdir(sample_dir))
            self.classes = class_names
            label = 0
            #Loop through data frame and get each image
            for img_folder in class_names:
                #Set class label 
                #Select folder and remove class number/space in name
                temp_img_folder = os.path.join(sample_dir,img_folder)
                for image in os.listdir(temp_img_folder):
                    #Check for correct size image
                    if image.startswith(str(image_size)):
                        if(image=='Thumbs.db'):
                            print('Thumb image') 
                        else:
                            img_file = os.path.join(temp_img_folder,image)
                            self.files.append({  # appends the images
                                    "img": img_file,
                                    "label": label
                                })
                            self.targets.append(label)
                label +=1

        else:  # test
            sample_dir = os.path.join(imgset_dir,'test')
            class_names = sorted(os.listdir(sample_dir))
            self.classes = class_names
            label = 0
            #Loop through data frame and get each image
            for img_folder in class_names:
                #Set class label 
                #Select folder and remove class number/space in name
                temp_img_folder = os.path.join(sample_dir,img_folder)
                for image in os.listdir(temp_img_folder):
                    if(image=='Thumbs.db'):
                        print('Thumb image') 
                    else:
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
        img = Image.open(img_file).convert('RGB')

        label_file = datafiles["label"]
        label = torch.tensor(label_file)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, label,index



class Toy_Dataset(Dataset):
    def __init__(self,directory,transform=None):  
        
       
        self.transform = transform
        self.images = datasets.ImageFolder(directory,transform=transform)
        
        self.targets = self.images.targets
        
        self.classes = self.images.classes
        
    def __getitem__(self, index):
        data, target = self.images[index]
        
        return data, target, index

    def __len__(self):
        return len(self.images)
    

class PRMIDataset(Dataset):
    def __init__(self, root, subset, species=None, transform=None):   
        assert subset in ['train', 'val', 'test'], \
            "Subset can only be 'train','val' or 'test'."
        self.root = root
        self.subset = subset
        self.transform = transform
        self.files = []
        self.img_lt = []
        self.lab_lt = []
        # default species: cotton, papaya, sunflower, switchgrass
        if species == None:
            self.species = ['Cotton_736x552_DPI150',
                            'Papaya_736x552_DPI150',
                            'Sunflower_640x480_DPI120',
                            'Switchgrass_720x510_DPI300']
        else:
            self.species = species
        self.classes = self.species
        
        # get the list of image file that contains root
        for item in self.species:
            lab_json = os.path.join(self.root, self.subset, 'labels_image_gt',
                                    (item+'_'+self.subset+'.json'))
            with open(lab_json) as f:
              data = json.load(f)
              for img in data:
                  if img['has_root'] == 1:
                      img_dir = os.path.join(self.root, self.subset, 'images',
                                             item, img['image_name'])
                      lab = img['crop']
                      self.img_lt.append(img_dir)
                      self.lab_lt.append(lab)
          
        # encode the species type into class label
        label_encoder = preprocessing.LabelEncoder()
        self.lab_lt = label_encoder.fit_transform(self.lab_lt)

        for item in zip(self.img_lt, self.lab_lt):
            self.files.append({
                    "img": item[0],
                    "label": item[1]
                    })

    def __len__(self):     
        return len(self.files)
    
    def __getitem__(self, index):      
        datafiles = self.files[index]
        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')
        label = datafiles["label"]
        if self.transform is not None:
            img = self.transform(img)        
        return img, label, index



class UCMerced_Index(Dataset):
    def __init__(self, root, split='train',transform=None, download=True):  
        
        self.transform = transform
        self.split = split
        self.images = geodatasets.UCMerced(root,split,transforms=transform,
                                           download=download)
        self.targets = self.images.targets        
        self.classes = self.images.classes
       
    def __getitem__(self, index):
        image, target = self.images._load_image(index)
                
        if self.transform is not None:
            to_pil = T.ToPILImage()
            image = to_pil(image)
            image = self.transform(image)
        return image, target, index

    def __len__(self):
        return len(self.images)

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
        img = Image.fromarray(np.uint8(img))
        label = datafiles["label"]       

        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, label, index


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
        #pdb.set_trace()
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