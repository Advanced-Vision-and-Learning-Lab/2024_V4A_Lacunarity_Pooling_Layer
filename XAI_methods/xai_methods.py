import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import EigenCAM
from PIL import Image
import torchvision.transforms as transforms
import pdb

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
        if fractal and model_name == "resnet18_lacunarity":
            target_layers = model.module.features[-1]
        elif fusion and model_name == "resnet18_lacunarity":
            target_layers = model.module.features[-1]
        elif model_name == "resnet18_lacunarity":
            target_layers = [model.module.layer4[-1]]
        elif model_name == "convnext_lacunarity":
            target_layers = [model.module.features[-1]]
        elif fusion and model_name == "densenet161_lacunarity":
            target_layers = [model.module.features.norm5 ]
        elif model_name == "densenet161_lacunarity":
            target_layers = [model.module.features[-1]]

        if parallel:
            target_layers = target_layers
        else:
            target_layers = [model.avgpool]
        
        
        if dataset == "PlantVillage":
            image_paths = [
                "Datasets/PlantVillage/plant_village_classification/Corn___Northern_Leaf_Blight/image (104).JPG",
                "Datasets/PlantVillage/plant_village_classification/Apple___Cedar_apple_rust/image (105).JPG",
                "Datasets/PlantVillage/plant_village_classification/Tomato___Late_blight/image (1022).JPG",
                "Datasets/PlantVillage/plant_village_classification/Squash___Powdery_mildew/image (1103).JPG",
                "Datasets/PlantVillage/plant_village_classification/Tomato___Bacterial_spot/image (1020).JPG",
                "Datasets/PlantVillage/plant_village_classification/Potato___Late_blight/image (124).JPG",
                "Datasets/PlantVillage/plant_village_classification/Corn___Cercospora_leaf_spot Gray_leaf_spot/image (108).JPG"
            ]
        elif dataset == "DeepWeeds":
            image_paths = ["Datasets/DeepWeeds/rangeland_weeds_australia/prickly_acacia/20170727-111748-3.jpg",
                "Datasets/DeepWeeds/rangeland_weeds_australia/parthenium/20170906-092043-1.jpg",
                "Datasets/DeepWeeds/rangeland_weeds_australia/siam_weed/20171113-061016-1.jpg",
                "Datasets/DeepWeeds/rangeland_weeds_australia/snake_weed/20170207-141531-0.jpg",
                 "Datasets/DeepWeeds/rangeland_weeds_australia/chinee_apple/20161207-110837-0.jpg"
            ]
        elif dataset == "LeavesTex":
            image_paths = ["Datasets/LeavesTex1200/LeavesTex1200/c20_018_a_w02.png",
                "Datasets/LeavesTex1200/LeavesTex1200/c11_002_a_w03.png",
                "Datasets/LeavesTex1200/LeavesTex1200/c06_017_a_w03.png",
                "Datasets/LeavesTex1200/LeavesTex1200/c04_017_a_w02.png"
            ]
            
        for i, image_path in enumerate(image_paths):
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
