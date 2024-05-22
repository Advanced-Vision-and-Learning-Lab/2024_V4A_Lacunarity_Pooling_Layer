import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import EigenCAM
from PIL import Image
import torchvision.transforms as transforms
import pdb
import random
import os

def get_attributions(Params, dataloaders, model):
    NumRuns = Params['Splits'][Params['Dataset']]
    fusion = Params["fusion"]
    fractal = Params["fractal"]
    parallel = Params["Parallelize"]
    model_name = Params["Model_name"]
    dataset = Params["Dataset"]
    
    for split in range(0, NumRuns):
        sub_dir = '{}/{}/{}/{}/{}/{}/Run_{}/'.format(Params['folder'],
                                                     Params["pooling_layer"],
                                                     Params["agg_func"],
                                                     Params['mode'],
                                                     Params['Dataset'],
                                                     Params['Model_name'],
                                                     split+1)

        # Specify target layers based on model and configuration
        if fractal and model_name == "resnet18":
            target_layers = model.module.features[-1]
        elif fusion and model_name == "resnet18":
            target_layers = model.module.features[-1]
        elif model_name == "resnet18":
            target_layers = [model.module.layer4[-1]]
        elif model_name == "convnext_tiny":
            target_layers = [model.module.features[-1]]
        elif fusion and model_name == "densenet161":
            target_layers = [model.module.features.norm5 ]
        elif model_name == "densenet161":
            target_layers = [model.module.features[-1]]

        if parallel:
            target_layers = target_layers
        else:
            target_layers = [model.avgpool]


        # Get a list of image paths dynamically from the root directory
        dataset = Params["Dataset"]
        root_dir = Params['data_dir'][dataset]
        image_paths = []

        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(subdir, file))
        
        #Choose upto 10 images
        num_images_to_select = min(len(image_paths), 10)
        selected_image_paths = random.sample(image_paths, num_images_to_select)
            
        for i, image_path in enumerate(selected_image_paths):
            # Load the original image and resize
            original_image = Image.open(image_path)
            original_image_resized = original_image.resize((224, 224))
            original_image_resized.save(sub_dir + f'original_image_{i}.jpg')
            
            # Convert resized image to numpy array
            img = np.array(original_image_resized)
            img = np.float32(img) / 255

            # Apply torchvision transform to convert to tensor and unsqueeze to add batch dimension
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            tensor = transform(img).unsqueeze(0)
            
            # Process the image to generate visualization
            cam = EigenCAM(model=model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=tensor)
            grayscale_cam = grayscale_cam[0, :]
            input_array = tensor[0].permute(1, 2, 0).numpy()
            visualization = show_cam_on_image(input_array, grayscale_cam, use_rgb=True)
            visualization_image = Image.fromarray(visualization, 'RGB')

            # Save the visualization
            visualization_image.save(sub_dir + f'visualization_image_{i}.jpg')
