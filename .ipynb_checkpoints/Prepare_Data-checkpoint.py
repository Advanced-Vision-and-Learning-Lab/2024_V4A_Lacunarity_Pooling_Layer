# -*- coding: utf-8 -*-
"""
Create datasets and dataloaders for models
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import pdb
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from pytorch_metric_learning import samplers
from Datasets.Split_Data import DataSplit
import itertools
import ssl
from sklearn.model_selection import StratifiedKFold
## PyTorch dependencies
import torch
## Local external libraries
from Datasets.Pytorch_Datasets import *
from Datasets.Get_transform import *
from Datasets import preprocess
from Datasets import loader
from barbar import Bar

def Compute_Mean_STD(trainloader):
    print('Computing Mean/STD')
    'Code from: https://stackoverflow.com/questions/60101240/finding-mean-and-standard-deviation-across-image-channels-pytorch'
    nimages = 0
    mean = 0.0
    var = 0.0
    for i_batch, batch_target in enumerate(Bar(trainloader)):
        batch = batch_target[0]
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        var += batch.var(2).sum(0)
   
    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)
    print()
    
    return mean, std


def Prepare_DataLoaders(Network_parameters, split,input_size=224, view_results = True):
    ssl._create_default_https_context = ssl._create_unverified_context
    
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']

    
    # Data augmentation and normalization for training
    # Just normalization and resize for test
    # Data transformations as described in:
    # http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf
    global data_transforms
    if Dataset == 'Synthetic_Gray':
        pass
    else:
        data_transforms = get_transform(Network_parameters, input_size=224)
    dataset_sampler = None

    if Dataset == 'FashionMNIST':
        train_dataset = FashionMNIST_Index(data_dir,train=True,transform=data_transforms['train'],
                                        download=True)
        y = train_dataset.targets
        indices = np.arange(len(y))
        indices = torch.as_tensor(indices)
        _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, 
                                                            test_size=0.1, 
                                                            stratify=y, random_state=42)
        
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        dataset_sampler = {'train': train_sampler, 'val': val_sampler, 'test': None}
        
        test_dataset = FashionMNIST_Index(data_dir,train=False,transform=data_transforms['test'],
                                        download=True)
            
       
    elif Dataset == 'BloodMNIST':
        train_dataset = BloodMNIST(data_dir, split='train', transform = data_transforms['train'], target_transform=None)
        test_dataset = BloodMNIST(data_dir, split='test', transform = data_transforms['test'], target_transform=None)
        val_dataset = BloodMNIST(data_dir, split='val', transform = data_transforms['test'], target_transform=None)

    elif Dataset == 'PneumoniaMNIST':
        train_dataset = PneumoniaMNIST(data_dir, split='train', transform = data_transforms['train'], target_transform=None)
        test_dataset = PneumoniaMNIST(data_dir, split='test', transform = data_transforms['test'], target_transform=None)
        val_dataset = PneumoniaMNIST(data_dir, split='val', transform = data_transforms['test'], target_transform=None)

    elif Dataset == 'OrganMNISTCoronal':
        train_dataset = OrganMNISTCoronal(data_dir, split='train', transform = data_transforms['train'], target_transform=None)
        test_dataset = OrganMNISTCoronal(data_dir, split='test', transform = data_transforms['test'], target_transform=None)
        val_dataset = OrganMNISTCoronal(data_dir, split='val', transform = data_transforms['test'], target_transform=None)

    elif Dataset == 'PlantLeaf':
        train_dataset = PlantLeaf(data_dir, split='train', transform = data_transforms['train'])
        test_dataset = PlantLeaf(data_dir, split='test', transform = data_transforms['test'])
        val_dataset = PlantLeaf(data_dir, split='val', transform = data_transforms['test'])

    elif Dataset == 'UCMerced': #See people also use .5, .5 for normalization
        train_dataset = UCMerced_Index(root = data_dir,split='train', transform = data_transforms['train'], download=True)
        test_dataset = UCMerced_Index(data_dir,split='test', transform = data_transforms['test'], download=True)
        val_dataset = UCMerced_Index(data_dir,split='val', transform = data_transforms['test'], download=True)

    elif Dataset == 'PRMI':
        # Call train and test
        train_dataset = PRMIDataset(data_dir,subset='train',transform=data_transforms['train'])
        test_dataset = PRMIDataset(data_dir, subset='test', transform=data_transforms['test'])
        val_dataset = PRMIDataset(data_dir, subset='val', transform=data_transforms['test'])

    elif Dataset == 'Synthetic_Gray' or Dataset == "Synthetic_RGB":
        
        # Create training, validation, and test datasets (no data augmentation for now)
        dataset = Toy_Dataset(data_dir,transform=data_transforms['val'])
    
        #Get train, val and test dataloaders
  
        split = DataSplit(dataset, shuffle=False,random_seed=split)
        train_loader, _ , _ = split.get_split(batch_size=Network_parameters['batch_size']['train'], 
                                                                num_workers=Network_parameters['num_workers'],
                                                                show_sample=False,
                                                                val_batch_size=Network_parameters['batch_size']['val'],
                                                                test_batch_size=Network_parameters['batch_size']['test'])
    
        #Compute mean/std for normalization
        mean,std = Compute_Mean_STD(train_loader)
        
        if 'Grayscale' in Dataset:
            data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(size=(200, 200)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]),
                'val': transforms.Compose([
                    transforms.Resize(size=(200, 200)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]),
            }
        
        else:
            data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(size=(200, 200)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]),
                'val': transforms.Compose([
                    transforms.Resize(size=(200, 200)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]),
            }
        
        #Create train/val/test loader based on mean and std
        split = DataSplit(dataset, shuffle=False,random_seed=split)
        train_loader, val_loader , test_loader = split.get_split(batch_size=Network_parameters['batch_size']['train'], 
                                                                num_workers=Network_parameters['num_workers'],
                                                                show_sample=False,
                                                                val_batch_size=Network_parameters['batch_size']['val'],
                                                                test_batch_size=Network_parameters['batch_size']['test'])
        # Create training and validation dataloaders, using validation set as testing for segmentation experiments
        dataloaders_dict = {'train': train_loader,'val': val_loader,'test': test_loader}

    elif Dataset == 'Kth_Tips': #Didn't use any data augmentation for initial experiments
        samples = ['a', 'b', 'c', 'd']
        # Set to 1 to train on 1, test 3; set to 2 to train on 2, test on 2;
        # set to 3 to train on 3 and test 1
        setting = 3
        
        sample_combos = list(itertools.combinations(samples, setting))
        train_setting = []
        test_setting = []
        for ii in range(0, len(sample_combos)):
            train_setting.append(list(sample_combos[ii]))
            test_setting.append(list(sorted(set(samples) - set(sample_combos[ii]))))

        train_val_dataset = KTH_TIPS_2b_data(data_dir,train=True,
                                         img_transform=data_transforms['train'],
                                         train_setting=train_setting[split])
        test_dataset = KTH_TIPS_2b_data(data_dir,train=False,
                                         img_transform=data_transforms['test'],
                                         test_setting=test_setting[split])
        
        indices = np.arange(len(train_val_dataset))
        y = train_val_dataset.targets

        # Use stratified split to balance training validation splits, set random state to be same for each encoding method
        _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, stratify=y, test_size=.1,
                                                                  random_state= 10)

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        dataset_sampler = {'train': train_sampler, 'val': val_sampler, 'test': None}

    elif Dataset == 'GTOS-mobile': #Need to create separate validation dataset from training
        # Create training and test datasets
        train_dataset = GTOS_mobile_single_data(data_dir, train = True,
                                           image_size=Network_parameters['resize_size'],
                                           img_transform=data_transforms['train']) 
        val_dataset = GTOS_mobile_single_data(data_dir, train = False,
                                           img_transform=data_transforms['test'])
        test_dataset = GTOS_mobile_single_data(data_dir, train = False,
                                           img_transform=data_transforms['test'])
    
    elif Dataset == "LeavesTex":
        dataset = LeavesTex1200(data_dir,transform=data_transforms['train'])
    
         #Create train/val/test loader based on mean and std
        split = DataSplit(dataset, shuffle=False,random_seed=split)
        train_loader, val_loader , test_loader = split.get_split(batch_size=Network_parameters['batch_size']['train'], 
                                                                num_workers=Network_parameters['num_workers'],
                                                                show_sample=False,
                                                                val_batch_size=Network_parameters['batch_size']['val'],
                                                                test_batch_size=Network_parameters['batch_size']['test'])
        # Create training and validation dataloaders, using validation set as testing for segmentation experiments

        dataloaders_dict = {'train': train_loader,'val': val_loader,'test': test_loader}

    elif Dataset == "Cassava":

        # Create training and test datasets
        train_dataset = Cassava(data_dir, train = True,
                                          image_size=Network_parameters['resize_size'],
                                          img_transform=data_transforms['train'])
        val_dataset = Cassava(data_dir, train = False,
                                          image_size=Network_parameters['resize_size'],
                                          img_transform=data_transforms['test']) 
        test_dataset = Cassava(data_dir, train = False,
                                          image_size=Network_parameters['resize_size'],
                                          img_transform=data_transforms['test']) 
            
    
    
    elif Dataset == "PlantVillage":
        loading = PlantVillage(root = data_dir)
        loader = loading.images

        #Split data
        loader.split(train = 0.7, val = 0.1, test = 0.2)
        train_dataset = loader.train_data
        train_dataset.as_torch_dataset()
        train_dataset.transform = data_transforms['train']

        test_dataset = loader.test_data
        test_dataset.as_torch_dataset()
        test_dataset.transform = data_transforms['test']

        val_dataset = loader.val_data
        val_dataset.as_torch_dataset()
        val_dataset.transform = data_transforms['test']

    elif Dataset == "DeepWeeds":
        loading = DeepWeeds(root = data_dir)
        loader = loading.images

        #Split data
        loader.split(train = 0.7, val = 0.1, test = 0.2)
        train_dataset = loader.train_data
        train_dataset.as_torch_dataset()
        train_dataset.transform = data_transforms['train']

        test_dataset = loader.test_data
        test_dataset.as_torch_dataset()
        test_dataset.transform = data_transforms['test']

        val_dataset = loader.val_data
        val_dataset.as_torch_dataset()
        val_dataset.transform = data_transforms['test']


    else:
        raise RuntimeError('{} Dataset not implemented'.format(Dataset)) 
    


    
    # if Dataset == "UCMerced":
    #     labels = test_dataset.targets
    #     classes = test_dataset.classes
    #     #m is the number of samples taken from each class
    #     m = 10
    #     #In our paper, batch_size for:
    #         #UCMerced - 210
    #         #EuroSAT - 100
    #         #MSTAR - 40
    #     batch_size = m*len(classes)
    #     sampler = samplers.MPerClassSampler(labels, m, batch_size, length_before_new_iter=100000)
    #     #retain sampler = None for 'train' and 'val' data splits
    #     dataset_sampler = {'train': None, 'test': sampler, 'val': None}
    #     Network_parameters["batch_size"]["test"] = batch_size

    # else:
    #     dataset_sampler = {'train': None, 'test': None, 'val': None}

    #Collate function is used only for EuroSAT and MSTAR
    #Compatible input size for Kornia augmentation
    if Dataset == "UCMerced":
    # Create training and test dataloaders
        image_datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                        batch_size=Network_parameters['batch_size'][x],
                                                        num_workers=Network_parameters['num_workers'],
                                                        pin_memory=Network_parameters['pin_memory'],
                                                        # sampler = dataset_sampler[x]
                                                        )
                                                        for x in ['train', 'val','test']}
        
    elif Dataset == "Kth_Tips":
        
        image_datasets = {'train': train_val_dataset, 'val': train_val_dataset, 'test': test_dataset}
        # Create training and test dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                           batch_size=Network_parameters['batch_size'][x],
                                                           sampler=dataset_sampler[x],
                                                           num_workers=Network_parameters['num_workers'],
                                                           pin_memory=Network_parameters['pin_memory'])
                            for x in ['train', 'val', 'test']}


    # Create training and test dataloaders
    #for FashionMNIST dataset
    elif Dataset == "FashionMNIST":
        image_datasets = {'train': train_dataset, 'val': train_dataset, 'test': test_dataset}
        # Create training and test dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                           batch_size=Network_parameters['batch_size'][x],
                                                           #sampler=dataset_sampler[x],
                                                           num_workers=Network_parameters['num_workers'],
                                                           pin_memory=Network_parameters['pin_memory'], shuffle=False,)
        for x in ['train', 'val', 'test']}
    
    elif Dataset == 'Synthetic_Gray' or Dataset=='LeavesTex':
            pass
    
    else:
        image_datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                        batch_size=Network_parameters['batch_size'][x],
                                                        num_workers=Network_parameters['num_workers'],
                                                        pin_memory=Network_parameters['pin_memory'],
                                                        shuffle=False,
                                                        )
                                                        for x in ['train', 'val','test']}
    

    return dataloaders_dict