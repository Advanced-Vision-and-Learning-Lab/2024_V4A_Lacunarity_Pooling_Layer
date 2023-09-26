# -*- coding: utf-8 -*-
"""
Functions to train/test model
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import time
import copy

## PyTorch dependencies
import torch
import torch.nn as nn
from torchvision import models

## Local external libraries
from barbar import Bar
from .pytorchtools import EarlyStopping
import pdb

def train_model(model, dataloaders, criterion, optimizer, device,
                num_epochs=25, scheduler=None):
    
    since = time.time()
    best_epoch = 0
    val_acc_history = []
    train_acc_history = []
    train_error_history = []
    val_error_history = []
    
    early_stopping = EarlyStopping(patience=10, verbose=True)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
   
    try:
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            
 
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode 
                else:
                    model.eval()   # Set model to evaluate mode
                
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data.
                for idx, (inputs, labels, index) in enumerate(Bar(dataloaders[phase])):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    index = index.to(device)
        
                    # zero the parameter gradients
                    optimizer.zero_grad()
                  
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        # Get model outputs and calculate loss
                        outputs = model(inputs)
                        
                        #Backward produces 2 losses
                        loss = criterion(outputs, labels.squeeze(1).long()).mean()
                     
                        _, preds = torch.max(outputs, 1)
        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                          
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds.data == labels.data)
            
                epoch_loss = running_loss / (len(dataloaders[phase].sampler))
                epoch_acc = running_corrects.double().cpu().numpy() / (len(dataloaders[phase].sampler))
                
                if phase == 'train':
                    if scheduler is not None:
                        scheduler.step()
                    train_error_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc)
    
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_epoch = epoch
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
    
                if phase == 'val':
                    valid_loss = epoch_loss
                    val_error_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc)
    
                print()
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)) 
               
             
           #Check for early stopping (end training)
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print()
                print("Early stopping")
                break
            
            if torch.isnan(torch.tensor(valid_loss)):
                print()
                print('Loss is nan')
                break
         
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best Val Acc: {:4f}'.format(best_acc))
        print()

    except:
        
        # load best model weights
        model.load_state_dict(best_model_wts)
        
        # Return losses as dictionary
        train_loss = train_error_history
        
        val_loss = val_error_history
     
        #Return training and validation information
        train_dict = {'best_model_wts': best_model_wts, 'val_acc_track': val_acc_history, 
                     'train_acc_track': train_acc_history, 
                      'train_error_track': train_loss,'best_epoch': best_epoch}
       
        print('Saved interrupt')
        return train_dict

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    # Return losses as dictionary
    train_loss = train_error_history
    
    val_loss = val_error_history
 
    #Return training and validation information
    train_dict = {'best_model_wts': best_model_wts, 'val_acc_track': val_acc_history, 
                  'val_error_track': val_loss,'train_acc_track': train_acc_history, 
                  'train_error_track': train_loss,'best_epoch': best_epoch}
    
    return train_dict

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
     
def test_model(dataloader,model,criterion,device,model_weights=None):
    #Initialize and accumalate ground truth, predictions, and image indices
    GT = np.array(0)
    Predictions = np.array(0)
    Index = np.array(0)
    
    running_corrects = 0
    running_loss = 0.0
    
    if model_weights is not None:
        model.load_state_dict(model_weights)
        
    model.eval()
    
    # Iterate over data
    print('Testing Model...')
    with torch.no_grad():
        for idx, (inputs, labels, index) in enumerate(Bar(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            index = index.to(device)
    
            #Run data through best model
            outputs = model(inputs)
         
            #Make model predictions
            _, preds = torch.max(outputs, 1)
            
            #Compute loss
            loss = criterion(outputs, labels.squeeze(1).long()).mean() ##########CHANGED

            #If test, accumulate labels for confusion matrix
            GT = np.concatenate((GT,labels.detach().cpu().numpy()),axis=None)
            Predictions = np.concatenate((Predictions,preds.detach().cpu().numpy()),axis=None)
            Index = np.concatenate((Index,index.detach().cpu().numpy()),axis=None)
            
            #Keep track of correct predictions
            running_corrects += torch.sum(preds == labels.data)
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
            
    epoch_loss = running_loss / (len(dataloader.sampler))
    test_acc = running_corrects.double() / (len(dataloader.sampler))
    
    test_loss = {'total': epoch_loss}
    
    print('Test Accuracy: {:4f}'.format(test_acc))
    
    test_dict = {'GT': GT[1:], 'Predictions': Predictions[1:], 'Index':Index[1:],
                 'test_acc': np.round(test_acc.cpu().numpy()*100,2),
                 'test_loss': test_loss}
    
    return test_dict

def lacunarity_global(x, eps = 10e-6):
    '''
    Lacunarity definition from Fast Unsupervised Seafloor Characterization
    in Sonar Imagery Using Lacunarity
    '''
    ##Convert input to tensor
    x_tensor = torch.tensor(x).unsqueeze(0).unsqueeze(0).float()
    squared_x_tensor = x_tensor ** 2
    n_pts = np.prod(np.asarray(x_tensor.shape))
    #Define sum pooling 
    gap_layer = nn.AdaptiveAvgPool2d(1)(x_tensor)
    
    #Compute numerator (n * sum of squared pixels) and denominator (squared sum of pixels)
    L_numerator = ((n_pts)**2) * (nn.AdaptiveAvgPool2d(1)(squared_x_tensor))
    L_denominator = (n_pts * gap_layer)**2

    #Lacunarity is L_numerator / L_denominator - 1
    L_torch = (L_numerator / L_denominator) - 1
    
    # Convert back to numpy array
    L_torch = L_torch.squeeze(0).squeeze(0)
    L_numpy = L_torch.detach().cpu().numpy()

    return L_numpy

    
def initialize_model(model_name, num_classes,feature_extract=False,
                     use_pretrained=False, channels = 3):
    
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    model_ft = None
    input_size = 0
  
    #Select backbone architecture
    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "resnet50_wide":
        model_ft = models.wide_resnet50_2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "resnet50_next":
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "densenet121":
        model_ft = models.densenet121(pretrained=use_pretrained,memory_efficient=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Sequential()
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "convnext":
        model_ft = models.convnext_base(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        if not(channels == 3):
            model_ft.features[0][0] = nn.Conv2d(channels, 128, kernel_size=(4,4),stride=(4,4))
        
        num_ftrs = model_ft.classifier[2].in_features
        model_ft.classifier[2] = nn.Linear(model_ft.classifier[2].in_features, num_classes)
        
        #Option 1: Replace first convolution layer to match input bands
        #model_ft.features[0][0] = nn.Conv2d(13,128,kernel_size=(4,4),stride=(4,4))
        
        #Option 2: Add 1x1 conv to reduce number of input channels to 3
        #model_ft = nn.Sequential(nn.Conv2d(13,3,kernel_size=(1,1)),model_ft)        
        input_size = 224

    elif model_name == "vit":
        model_ft = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)
        if not(channels == 3):
            model_ft.conv_proj = nn.Conv2d(13, 768, kernel_size=(16,16),stride=(16,16)) 
        num_ftrs = model_ft.heads.head.in_features
        model_ft.heads.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "Swin":
        model_ft = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.head.in_features
        model_ft.head = nn.Linear(num_ftrs, num_classes)

    elif model_name == "retinaNet":
        model_ft = models.detection.retinanet_resnet50_fpn_v2()
    
    else:
        raise RuntimeError('{} not implemented'.format(model_name))
    
    return model_ft, input_size

