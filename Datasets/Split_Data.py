# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:21:47 2021
Code modified from https://palikar.github.io/posts/pytorch_datasplit/
Ensure balanced split among classes
@author: jpeeples
"""

import logging
from functools import lru_cache

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
import matplotlib.pyplot as plt

import numpy as np
from numpy import random
import pdb


class DataSplit:

    def __init__(self, train_dataset,val_dataset,test_dataset, test_train_split=0.8, val_train_split=0.1, 
                 random_seed=0,shuffle=False,stratify=True):
        
        #self.dataset = dataset
        self.stratify = stratify
        self.random_seed = random_seed
        dataset_size = len(train_dataset)
        labels = train_dataset.targets
        
        self.indices = list(range(dataset_size))
        test_split = int(np.floor(test_train_split * dataset_size))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(self.indices)

        if stratify:
            _,_,_,_,train_indices,self.test_indices = train_test_split(labels,labels,self.indices,stratify=labels)
        else:
            train_indices, self.test_indices = self.indices[:test_split], self.indices[test_split:]
        
        train_size = len(train_indices)
        validation_split = int(np.floor((1 - val_train_split) * train_size))

        if stratify:
            self.train_indices, self.val_indices = train_indices[ : validation_split], train_indices[validation_split:]
        else:
            _,_,_,_,self.train_indices,self.val_indices = train_test_split(labels,labels,self.train_indices,stratify=labels[train_indices])

        # self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset,[.7,.1,.2])
        self.train_dataset, self.val_dataset, self.test_dataset = train_dataset, val_dataset, test_dataset
        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_train_split_point(self):
        return len(self.train_sampler) + len(self.val_indices)

    def get_validation_split_point(self):
        return len(self.train_sampler)
    
    def seed_worker(self,worker_id):
        #lambda x: np.random.seed((torch.initial_seed() + x) % (2 ** 32))
        # worker_seed = torch.initial_seed() % 2**32
        np.random.seed(0)
        random.seed(0)

    @lru_cache(maxsize=4)
    def get_split(self, batch_size=4, num_workers=4, show_sample=True, val_batch_size=4,
                  test_batch_size=4,collate_fn = None):
        logging.debug('Initializing train-validation-test dataloaders')
        
        if val_batch_size is None:
            self.train_loader = self.get_train_loader(batch_size=batch_size, 
                                                      num_workers=num_workers,
                                                      collate_fn = collate_fn)
            self.val_loader = self.get_validation_loader(batch_size=batch_size, 
                                                         num_workers=num_workers,
                                                         collate_fn = collate_fn)
            self.test_loader = self.get_test_loader(batch_size=batch_size, 
                                                    num_workers=num_workers,
                                                    collate_fn = collate_fn)
        else:
            self.train_loader = self.get_train_loader(batch_size=batch_size, 
                                                      num_workers=num_workers,
                                                      collate_fn = collate_fn)
            self.val_loader = self.get_validation_loader(batch_size=val_batch_size, 
                                                         num_workers=num_workers,
                                                         collate_fn = collate_fn)
            self.test_loader = self.get_test_loader(batch_size=test_batch_size,
                                                    num_workers=num_workers,
                                                    collate_fn = collate_fn)
        
        # visualize some training images
        if show_sample:
            sample_loader = self.train_loader
            data_iter = iter(sample_loader)
            images, labels, _ = data_iter.next()
            X = images.numpy().transpose([0, 2, 3, 1])
            self.plot_images(X, labels)

        return self.train_loader, self.val_loader, self.test_loader

    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size=4, num_workers=4,collate_fn = None):
        logging.debug('Initializing train dataloader')
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, 
                                                        sampler=self.train_sampler, shuffle=False, 
                                                        num_workers=num_workers, worker_init_fn=self.seed_worker,
                                                        collate_fn = collate_fn)
        return self.train_loader

    @lru_cache(maxsize=4)
    def get_validation_loader(self, batch_size=4, num_workers=4,collate_fn = None):
        logging.debug('Initializing validation dataloader')
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, 
                                                      sampler=self.val_sampler, shuffle=False, 
                                                      num_workers=num_workers, worker_init_fn=self.seed_worker,
                                                      collate_fn = collate_fn)
        return self.val_loader

    @lru_cache(maxsize=4)
    def get_test_loader(self, batch_size=4, num_workers=4,collate_fn = None):
        logging.debug('Initializing test dataloader')
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, 
                                                       sampler=self.test_sampler, shuffle=False, 
                                                       num_workers=num_workers, worker_init_fn=self.seed_worker,
                                                       collate_fn = collate_fn)
        return self.test_loader
    
    def plot_images(self,images, cls_true, cls_pred=None):
        """
        Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
        Modified code from: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
        """
        
        size = int(np.sqrt(images.shape[0]))
        fig, axes = plt.subplots(size, size)
        label_names = self.dataset.classes
        for i, ax in enumerate(axes.flat):
            # plot img
            img = ax.imshow(images[i, :, :, :], interpolation='spline16',cmap='pink')
            plt.colorbar(img,ax=ax)
    
            # show true & predicted classes
            cls_true_name = label_names[cls_true[i]]
            if cls_pred is None:
                xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
            else:
                cls_pred_name = label_names[cls_pred[i]]
                xlabel = "True: {0}\nPred: {1}".format(
                    cls_true_name, cls_pred_name
                )
            ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])
    
        plt.show()
    
    
