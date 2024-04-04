
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:19:34 2019
Generate confusion matrices for results
@author: jpeeples
"""
## Python standard libraries
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pdb

def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues,
                         show_percent=True,ax=None,fontsize=12):
   """
   This function prints and plots the confusion matrix.
   Normalization can be applied by setting normalize=True.
   """
   if normalize:
       cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
       print("Normalized confusion matrix")
   else:
       print('Confusion matrix, without normalization')
#   print(cm)
   
   #Generate new figure if needed
   if ax is None:
       fig, ax = plt.subplots()
        
   if show_percent:
       cm_percent = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
       im = ax.imshow(cm_percent,interpolation='nearest',cmap=cmap,vmin=0,vmax=100)
       #plt.title(title)
       cb = plt.colorbar(im,fraction=0.046,pad=0.04)
       cb.ax.tick_params(labelsize=fontsize)
   else:
       im = ax.imshow(cm,interpolation='nearest',cmap=cmap)
       plt.title(title)
       cb = plt.colorbar(im,fraction=0.046,pad=0.04)
       cb.ax.tick_params(labelsize=fontsize)
       
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, fontsize = fontsize)
   plt.yticks(tick_marks, classes, fontsize = fontsize)
   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if show_percent:
            s = str(format(cm[i, j], fmt) + '\n' + 
                    '(' + format(cm_percent[i,j],'.2f')+'%)')
        else:
            s = format(cm[i, j], fmt)
        ax.text(j, i, s,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=fontsize)
       
   ax.set(xticks=np.arange(len(classes)),
                   yticks=np.arange(len(classes)),
                   xticklabels=classes,
                   yticklabels=classes)
                   #label="True label",
                   #xlabel="Predicted label")
   
   ax.set_ylim((len(classes) - 0.5, -0.5))
   plt.setp(ax.get_xticklabels(), rotation=45)
   #plt.ylabel('True label')
   #plt.xlabel('Predicted label')
   plt.tight_layout()
       
def plot_avg_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues,
                              show_percent=True,
                              ax=None,
                              fontsize=9):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute average CM values
    std_cm = np.int64(np.ceil(np.std(cm, axis=2)))
    cm = np.int64(np.ceil(np.mean(cm, axis=2)))

    # Generate new figure if needed
    if ax is None:
        fig, ax = plt.subplots()

    if show_percent:
        cm_percent = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_percent_std = 100 * std_cm.astype('float') / (std_cm.sum(axis=1)[:, np.newaxis] + 10e-6)
        im = ax.imshow(cm_percent, interpolation='nearest', cmap=cmap, vmin=0, vmax=100)
        cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=fontsize)

    else:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=fontsize)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    classes =  [
    'Apple Apple_scab',
    'Apple Black_rot',
    'Apple Cedar_apple_rust',
    'Apple healthy',
    'Background_without_leaves',
    'Blueberry healthy',
    'Cherry Powdery_mildew',
    'Cherry healthy',
    'Corn Cercospora_leaf_spot',
    'Corn Common_rust',
    'Corn Northern_Leaf_Blight',
    'Corn healthy',
    'Grape Black_rot',
    'Grape Esca',
    'Grape Leaf_blight',
    'Grape healthy',
    'Orange Haunglongbing',
    'Peach Bacterial_spot',
    'Peach healthy',
    'Pepper, bell Bacterial_spot',
    'Pepper, bell healthy',
    'Potato Early_blight',
    'Potato Late_blight',
    'Potato healthy',
    'Raspberry healthy',
    'Soybean healthy',
    'Squash Powdery_mildew',
    'Strawberry Leaf_scorch',
    'Strawberry healthy',
    'Tomato Bacterial_spot',
    'Tomato Early_blight',
    'Tomato Late_blight',
    'Tomato Leaf_Mold',
    'Tomato Septoria_leaf_spot',
    'Tomato Spider_mites',
    'Tomato Target_Spot',
    'Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_mosaic_virus',
    'Tomato healthy'
]

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=fontsize)  # Set font size initially
    plt.yticks(tick_marks, classes, fontsize=fontsize)  # Set font size initially
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Adjust font size dynamically based on the length of class names
    max_class_length = max([len(c) for c in classes])
    if max_class_length > 5:
        fontsize = 6  # Adjust this value as needed

    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes,
           yticklabels=classes)
    ax.set_ylim((len(classes) - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
