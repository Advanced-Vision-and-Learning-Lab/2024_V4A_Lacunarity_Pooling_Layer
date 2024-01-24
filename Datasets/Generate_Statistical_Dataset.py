# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 12:54:53 2021
Create synthetic dataset for structural and statistical textures
@author: jpeeples
"""

import os
from PIL import Image as im
from torch.utils.data import Dataset
from natsort import natsorted
import scipy.io as sio
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from os import path
import random
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal, norm
from skimage import img_as_bool
from skimage.transform import resize
import scipy.stats as stats

import pdb
import torch



#TBD work on expansion beyond 3
def inv_sigmoid(values):    
    return np.log(values/(1-values))

def Generate_Stats_Textures(img_dir,num_imgs,sigma=.1,random_seed=1,spatial=3,
                            grayscale=True,plot=False):
    
    #Set random seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    # mus = [np.array([0, 1, 0]), np.array([20, 1, 0]), np.array([-3, 3, 0])]
    if grayscale:
        sigma = np.sqrt(sigma)
        mus = [np.array([64]), np.array([128]), np.array([192])]
        covs = [np.array([sigma]), 
                np.array([sigma]), 
                np.array([sigma])]
    else:            
        mus = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        covs = [np.array([[1, 2, 1], [2, 1, 1], [2, 1, 1]]), 
                np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]), 
                np.array([[10, 1, 1], [1, 0.3, 1], [1, 0.3, 1]])]

    # pis = np.array([0.5, 0, 0.5])
    list_pis = [np.array([0.5, 0, 0.5]),np.array([1/3,1/3,1/3]),np.array([0,1,0])]
    # list_pis = [np.array([0.1, 0.8, 0.1]),np.array([1/3,1/3,1/3]),np.array([0.45,.1,0.45])]
    # list_pis = [np.array([0.4, 0, 0.6]),np.array([.2,.2,.6]),np.array([0,.9,.1])]
    # pis = np.array([1/3,1/3,1/3])
    # pis = np.array([0,1,0])
    # acc_pis = [np.sum(pis[:i]) for i in range(1, len(pis) + 1)]
    # assert np.isclose(acc_pis[-1], 1)
      
    # pdb.set_trace()
    num_pts = num_imgs * spatial**2
    
    labels = ['Stat_1','Stat_2','Stat_3']
    stat_samples = []
    
    label_count = 0
    color_labels = []
    for label in labels:
        samples = []
        pis = list_pis[label_count]
        acc_pis = [np.sum(pis[:i]) for i in range(1, len(pis) + 1)]
        assert np.isclose(acc_pis[-1], 1)
        for i in range(num_pts):
            # sample uniform
            r = np.random.uniform(0, 1)
           
            # select gaussian
            k = 0
            for i, threshold in enumerate(acc_pis):
                if r < threshold:
                    k = i
                    break
    
            selected_mu = mus[k]
            selected_cov = covs[k]
    
            if grayscale:
                x_normal = np.random.normal(selected_mu,selected_cov,1)
                samples.append(x_normal)
            else:
                # sample from selected gaussian
                lambda_, gamma_ = np.linalg.eig(selected_cov)
        
                dimensions = len(lambda_)
                
                # sampling from normal distribution
                y_s = np.random.uniform(0, 1, size=(dimensions * 1, 3))
                x_normal = np.mean(inv_sigmoid(y_s), axis=1).reshape((-1, dimensions))
                
                # transforming into multivariate distribution
                x_multi = (x_normal * lambda_) @ gamma_ + selected_mu
                samples.append(x_multi.tolist()[0])
            color_labels.append(label_count)
        stat_samples.append(samples)
        label_count += 1
    
    r_data = np.asarray(stat_samples[0])
    g_data = np.asarray(stat_samples[1])
    b_data = np.asarray(stat_samples[2])
    color_labels = np.asarray(color_labels)
    
    if plot:
        fig = plt.figure()
        color_distribution = np.concatenate((r_data,g_data,b_data),axis=0)
        if grayscale:
            ax = fig.add_subplot(111)
            rv = norm(loc=mus[0],scale=covs[0])
            rv1 = norm(loc=mus[1],scale=covs[1])
            rv2 = norm(loc=mus[2],scale=covs[2])
            x = np.arange(0,255,.1)
            plt.plot(x, rv.pdf(x), x, rv1.pdf(x), x, rv2.pdf(x))
        else:
            ax = fig.add_subplot(111,projection='3d')
            ax.scatter(color_distribution[:,0],color_distribution[:,1],color_distribution[:,2],
                       c=color_labels)
    
    #Create prototypes for shapes
    cross_prototype = np.array([[0,1,0],[1,1,1],[0,1,0]])
    checkerboard_prototype = np.array([[1,0,1],[0,1,0],[1,0,1]])
    stripe_prototype = np.array([[0,1,0],[0,1,0],[0,1,0]])
    
    cross_prototype = img_as_bool(resize(cross_prototype.astype(bool),(spatial,spatial))).astype(float)
    checkerboard_prototype = img_as_bool(resize(checkerboard_prototype.astype(bool),(spatial,spatial))).astype(float)
    stripe_prototype = img_as_bool(resize(stripe_prototype.astype(bool),(spatial,spatial))).astype(float)
    
    #Reshape RGB information and contentate
    r_data = r_data.reshape((-1,1,spatial,spatial))
    b_data = b_data.reshape((-1,1,spatial,spatial))
    g_data = g_data.reshape((-1,1,spatial,spatial))
    
    if grayscale:
        r_data = r_data.reshape(num_imgs,spatial,spatial,1)
        g_data = g_data.reshape(num_imgs,spatial,spatial,1)
        b_data = b_data.reshape(num_imgs,spatial,spatial,1)
    else:
        r_data = r_data.reshape(num_imgs,spatial,spatial,3)
        g_data = g_data.reshape(num_imgs,spatial,spatial,3)
        b_data = b_data.reshape(num_imgs,spatial,spatial,3)
    
    #Expand prototypes to be B x M x N x 1
    cross_prototype = np.repeat(np.expand_dims(cross_prototype,axis=(0,-1)),repeats=num_imgs,axis=0)
    checkerboard_prototype = np.repeat(np.expand_dims(checkerboard_prototype,axis=(0,-1)),repeats=num_imgs,axis=0)
    stripe_prototype = np.repeat(np.expand_dims(stripe_prototype,axis=(0,-1)),repeats=num_imgs,axis=0)
    
    #Use prototype masks to generate images with different shapes and distributions
    prototype_names = ['Cross', 'Checkerboard', 'Stripe']
    color_dist_names = ['Stat_1', 'Stat_2', 'Stat_3']
    prototypes = {0: cross_prototype, 1: checkerboard_prototype, 2: stripe_prototype}
    color_dist = {0: r_data, 1: g_data, 2: b_data}

    if plot == True:
        fig, ax = plt.subplots(3,3,figsize=(12,6))
    
    for prototype in prototypes:
        for color in color_dist:
            current_prototype = prototypes[prototype]*color_dist[color]
            label_name = "{}_{}".format(color_dist_names[color],prototype_names[prototype])
           
            #Create directory and subdirectory for each type
            sub_location = img_dir + '/' + label_name + '/'
           
            if not os.path.exists(sub_location):
                os.makedirs(sub_location)
            
            #Loop through each image and save png file to load for Pytorch
            for img in range(0,num_imgs):
                if grayscale:
                    temp_img = current_prototype[img][:,:,0]
                else:
                    temp_img = current_prototype[img]
                # temp_img = im.fromarray(temp_img.astype(np.uint8)).resize((224,224))
                # print(np.mean(temp_img))
                if img == 0 and plot == True:
                    # pdb.set_trace()
                    im_fig = ax[prototype,color].imshow(temp_img,cmap='gray',vmin=0,vmax=255)
                    ax[prototype,color].set_title(label_name)
                    ax[prototype,color].axis('off')
                
                temp_img = im.fromarray(temp_img.astype(np.uint8))
                temp_img.save("{}{}_{}.png".format(sub_location,label_name,img))
                
            print('Finished generating {}'.format(label_name))
    

if __name__ == '__main__':
    
    list_sigma = [.1]
    random_seed = 1
    num_imgs = 100
    grayscale = True
    spatial = 224
    
    for sigma in list_sigma:
        if grayscale:
            img_path = 'Synthetic_Gray_Sigma_{}'.format(sigma)
        else:
            img_path = 'Synthetic_RGB_Sigma_{}'.format(sigma)
        Generate_Stats_Textures(img_path,num_imgs,sigma=sigma,spatial=spatial,
                                random_seed=random_seed,grayscale=False,plot=False)
    