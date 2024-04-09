# Lacunarity Pooling Layers for Plant Image Texture Analysis:
**Lacunarity Pooling Layers for Plant Image Texture Analysis**

_Akshatha Mohan and Joshua Peeples_

![Fig1_Workflow](Images/overviewimage.png)

Note: If this code is used, cite it: Akshatha Mohan and Joshua Peeples. 
[`Zendo`](https://zenodo.org/record/8023959). https://doi.org/10.5281/zenodo.8023959
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8023959.svg)](https://doi.org/10.5281/zenodo.8023959)

[`IEEE Xplore (IGARRS)`](TBD)

[`arXiv`](TBD)

[`BibTeX`](TBD)

In this repository, we provide the paper and code for the "Quantitative Analysis of Primary Attribution Explainable Artificial Intelligence Methods for Remote Sensing Image Classification."

## Installation Prerequisites

This code uses python, pytorch, quantus, and captum. 
Please use [`Pytorch's website`](https://pytorch.org/get-started/locally/) to download necessary packages.

## Demo

Run `demo.py` in Python IDE (e.g., Spyder) or command line. 

## Main Functions

The XAI Analyis runs using the following functions. 

1. Intialize model  

```model, input_size = intialize_model(**Parameters)```

2. Prepare dataset(s) for model

 ```dataloaders_dict = Prepare_Dataloaders(**Parameters)```

3. Train model 

```train_dict = train_model(**Parameters)```

4. Test model

```test_dict = test_model(**Parameters)```


## Parameters
The parameters can be set in the following script:

```Demo_Parameters.py```

## Inventory

```
https://github.com/Peeples-Lab/XAI_Analysis

└── root dir
	├── demo.py   //Run this. Main demo file.
	├── Demo_Parameters.py // Parameters file for demo.
	├── Prepare_Data.py  // Load data for demo file.
	├── View_Results.py // Run this after demo to view saved results.
    	├── Datasets
		├── Get_transform.py // Transforms applied on test, train, val dataset splits
		├── Pytorch_Datasets.py // Return Index for Pytorch datasets
		├── Pytorch_Datasets_Names.py // Return names of classes in each dataset
		├── Split_Data.py // Returns data splits for train, test and validation
	└── Utils  //utility functions
		├── Compute_FDR.py  // Compute Fisher Score
		├── Confusion_mats.py  // Create and plot confusion matrix.
    		├── Generating_Learning_Curves.py  // Plot training and validation accuracy and error measures.
    		├── Generate_TSNE_visual.py  // Create TSNE visual for results.
    		├── Network_functions.py  // Contains functions to initialize, train, and test model. 
    		├── pytorchtools.py // Function for early stopping.
		├── CustomNN.py // Contains the shallow neural network and densenet161 model implementation
		├── Lacunarity_Pooling_Layer.py // Implemented all the lacunarity calculation variants. (Base, DBC, multi-scale)
		├── Fusion_Model.py // Contains fractal pooling and fusion model
    		├── Save_Results.py  // Save results from demo script.
	└── XAI_Methods  // XAI functions
		├── xai_methods.py //Function to visualize EigenCAM on feature maps
```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) 
file in the root directory of this source tree.

This product is Copyright (c) 2023 A. Mohan and J. Peeples. All rights reserved.

## <a name="CitingLacunarity"></a>Citing Lacunarity Pooling Layers for Plant Image Texture Analysis

If you use the code, please cite the following 
reference using the following entry.

**Plain Text:**

A. Mohan and J. Peeples, "Quantitative Analysis of Primary Attribution Explainable Artificial Intelligence Methods for Remote Sensing Image Classification,"  in 2023 IEEE International Geoscience and Remote Sensing Symposium IGARSS, pp. 950-953. IEEE, 2023

**BibTex:**
```
@inproceedings{mohan2024lacunarity,
  title={Lacunarity Pooling Layers for Plant Image Texture Analysis},
  author={Mohan, Akshatha and Peeples, Joshua},
  booktitle={2024 Agriculture-vision IEEE/CVF CVPR 2024},
  pages={950-953},
  year={2024},
  organization={CVPR}
}

```
