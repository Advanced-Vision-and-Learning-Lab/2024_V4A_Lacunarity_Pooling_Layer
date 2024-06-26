# -*- coding: utf-8 -*-
"""
Main script for Lacunarity experiments
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Demo_Parameters import Parameters
from Utils.Save_Results import save_results
from Prepare_Data import Prepare_DataLoaders
from Utils.Network_functions import initialize_model, train_model, test_model
import os
import pdb

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#Turn off plotting
plt.ioff()


def main(Params):
  
   # Name of dataset
   Dataset = Params['Dataset'] 
  
   # Model(s) to be used
   model_name = Params['Model_name']
  
   # Number of classes in dataset
   num_classes = Params['num_classes'][Dataset]
  
   # Number of runs and/or splits for dataset
   numRuns = Params['Splits'][Dataset]
  
   # Detect if we have a GPU available
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


   print('Starting Experiments...')
  
   for split in range(0, numRuns):
       #Set same random seed based on split and fairly compare each embedding approach
       torch.manual_seed(split)
       np.random.seed(split)
       np.random.seed(split)
       torch.cuda.manual_seed(split)
       torch.cuda.manual_seed_all(split)
       torch.manual_seed(split)


       # Create training and validation dataloaders
       print("Initializing Datasets and Dataloaders...")
       dataloaders_dict = Prepare_DataLoaders(Params, split)        


       # Initialize the histogram model for this run
       model_ft, input_size = initialize_model(model_name, num_classes, dataloaders_dict, Params,
                                               aggFunc = Params["agg_func"])


       # Send the model to GPU if available, use multiple if available
       if torch.cuda.device_count() > 1:
           print("Using", torch.cuda.device_count(), "GPUs!")
           model_ft = nn.DataParallel(model_ft)
       
       model_ft = model_ft.to(device)

      # Print number of trainable parameters
       num_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)      
       print("Number of parameters: %d" % (num_params))     
       optimizer_ft = optim.Adam(model_ft.parameters(), lr=Params['lr'])
  
       #Loss function
       criterion = nn.CrossEntropyLoss()   
       scheduler = None


       # Train and evaluate
       train_dict = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, patience=Params['earlystoppping'],
                                 num_epochs=Params['num_epochs'],
                                 scheduler=scheduler)
       test_dict = test_model(dataloaders_dict['test'], model_ft, criterion,
                               device, model_weights = train_dict['best_model_wts'])




       # Save results
       if (Params['save_results']):
           #Delete previous dataloaders and training/validation data without data augmentation
           save_results(train_dict, test_dict, split, Params,
                         num_params,model_ft)
          
           del train_dict, test_dict, model_ft
           torch.cuda.empty_cache()


       print('**********Run ' + str(split + 1) + model_name + ' Finished**********')
     


def parse_args():
   parser = argparse.ArgumentParser(description='Run Angular Losses and Baseline experiments for dataset')
   parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction,
                       help='Save results of experiments(default: True)')
   parser.add_argument('--folder', type=str, default='Saved_Models',
                       help='Location to save models')
   parser.add_argument('--kernel', type=int, default=None,
                       help='Input kernel size')
   parser.add_argument('--stride', type=int, default=None,
                       help='Input stride size')
   parser.add_argument('--padding', type=int, default=0,
                       help='Input padding size')
   parser.add_argument('--num_levels', type=int, default=2,
                       help='Input number of levels')
   parser.add_argument('--pooling_layer', type=int, default=5,
                       help='pooling layer selection: 1:max, 2:avg, 3:L2, 4:fractal, 5:Base_Lacunarity, 6:MS_Lacunarity, 7:DBC_Lacunarity')
   parser.add_argument('--agg_func', type=int, default=1,
                       help='agg func: 1:global, 2:local')
   parser.add_argument('--data_selection', type=int, default=3,
                       help='Dataset selection: 1:LeavesTex1200, 2:PlantVillage, 3:DeepWeeds')
   parser.add_argument('--feature_extraction', default=True, action=argparse.BooleanOptionalAction,
                       help='Flag for feature extraction. False, train whole model. True, only update \
                        fully connected/encoder parameters (default: True)')
   parser.add_argument('--use_pretrained', default=True, action=argparse.BooleanOptionalAction,
                       help='Flag to use pretrained model from ImageNet or train from scratch (default: True)')
   parser.add_argument('--xai', default=False, action=argparse.BooleanOptionalAction,
                       help='enables xai interpretability')
   parser.add_argument('--earlystoppping', type=int, default=10,
                       help='early stopping for training')
   parser.add_argument('--train_batch_size', type=int, default=128,
                       help='input batch size for training (default: 128)')
   parser.add_argument('--val_batch_size', type=int, default=128,
                       help='input batch size for validation (default: 512)')
   parser.add_argument('--test_batch_size', type=int, default=128,
                       help='input batch size for testing (default: 256)')
   parser.add_argument('--num_epochs', type=int, default=1,
                       help='Number of epochs to train each model for (default: 50)')
   parser.add_argument('--resize_size', type=int, default=256,
                       help='Resize the image before center crop. (default: 256)')
   parser.add_argument('--lr', type=float, default=0.01,
                       help='learning rate (default: 0.01)')
   parser.add_argument('--model', type=str, default='resnet18',
                       help='backbone architecture to use (default: 0.01). Model choices = resnet18, densenet161, convnext_tiny.fb_in22k')
   parser.add_argument('--use-cuda', action='store_true', default=True,
                       help='enables CUDA training')
   args = parser.parse_args()
   return args


if __name__ == "__main__":
   args = parse_args()
   use_cuda = args.use_cuda and torch.cuda.is_available()
   device = torch.device("cuda" if use_cuda else "cpu")
   params = Parameters(args)
   main(params)

