
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pdb

from pytorch_grad_cam.utils.image import show_cam_on_image
import random
import warnings
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torch
import tqdm
from PIL import Image



def transform_list(saved_list):
    'Transform list to numpy array'
    saved_list = np.concatenate(saved_list,axis=0)
    return saved_list

def get_attributions(dataloaders, dataset, model, device, Params, parallel=False):
    
    target_layers = [model.avgpool]
    images, labels, index = iter(dataloaders).__next__()
    input = images
    cam = EigenCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(input, grayscale_cam, use_rgb=True)
    # Convert the visualization from numpy array to PIL Image
    visualization_image = Image.fromarray(visualization, 'RGB')
    # Save the PIL Image to a file
    visualization_image.save("visualization_image.jpg")

