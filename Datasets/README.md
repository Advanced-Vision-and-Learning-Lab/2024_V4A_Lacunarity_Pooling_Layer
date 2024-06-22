# Downloading datasets:

Note: Due to the size of the datasets, the images were not 
upload to the repository. Please follow the following instructions
to ensure the code works. If any of these datasets are used,
please cite the appropiate sources (papers, repositories, etc.) as mentioned
on the webpages and provided here.

## LeavesTex1200 dataset [[`BibTeX`](https://github.com/Advanced-Vision-and-Learning-Lab/2024_V4A_Lacunarity_Pooling_Layer/tree/main/Datasets#citing-leavestex1200)]
To download the LeavesTex1200 dataset. Download the <a href="http://scg-turing.ifsc.usp.br/data/bases/LeavesTex1200.zip">download.zip</a>.
The structure of the `LeavesTex1200` folder is as follows:
```
└── root dir
    ├── LeavesTex1200 //Contains folders of images for each class.
    	├── Images
``` 

## <a name="CitingLeavesTex1200"></a>Citing LeavesTex1200
If you use this dataset in your research, please cite the following paper:
<a href="https://doi.org/10.1002/ima.20201"> Plant leaf identification using Gabor wavelets </a>


**BibTex:**
```
@article{casanova2009plant,
  title={Plant leaf identification using Gabor wavelets},
  author={Casanova, Dalcimar and de Mesquita S{\'a} Junior, Jarbas Joaci and Bruno, Odemir Martinez},
  journal={International Journal of Imaging Systems and Technology},
  volume={19},
  number={3},
  pages={236--243},
  year={2009},
  publisher={Wiley Online Library}
}
```

## PlantVillage dataset [[`BibTeX`](https://github.com/Advanced-Vision-and-Learning-Lab/2024_V4A_Lacunarity_Pooling_Layer/tree/main/Datasets#citing-plantvillage)]
To download the PlantVillage dataset. Download the dataset through <a href="https://github.com/Project-AgML/AgML">AgML</a>. If you use AgML in your research or project, please cite it as shown <a href="https://github.com/Advanced-Vision-and-Learning-Lab/2024_V4A_Lacunarity_Pooling_Layer/tree/main/Datasets#citing-agml">below</a>. 

``` python
import agml
loader = agml.data.AgMLDataLoader('plant_village_classification', dataset_path="Datasets/PlantVillage")
```

The structure of the `PlantVillage` folder is as follows:
```
└── root dir
    ├── PlantVillage
    	├── plant_village_classification
    		├── Classes
``` 

## <a name="CitingPlantVillage"></a>Citing PlantVillage
If you use this dataset in your research, please cite the following paper:
<a href="https://doi.org/10.48550/arXiv.1511.08060"> An open access repository of images on plant health to enable the development of mobile disease diagnostics </a>

**BibTex:**
```
@article{hughes2015open,
  title={An open access repository of images on plant health to enable the development of mobile disease diagnostics},
  author={Hughes, David and Salath{\'e}, Marcel and others},
  journal={arXiv preprint arXiv:1511.08060},
  year={2015}
}
```

## DeepWeeds dataset [[`BibTeX`](https://github.com/Advanced-Vision-and-Learning-Lab/2024_V4A_Lacunarity_Pooling_Layer/tree/main/Datasets#citing-deepweeds)]
To download the DeepWeeds dataset. Download the dataset through <a href="https://github.com/Project-AgML/AgML">AgML</a>. If you use AgML in your research or project, please cite it as shown <a href="https://github.com/Advanced-Vision-and-Learning-Lab/2024_V4A_Lacunarity_Pooling_Layer/tree/main/Datasets#citing-agml">below</a>. 

``` python
import agml
loader = agml.data.AgMLDataLoader('rangeland_weeds_australia', dataset_path="Datasets/DeepWeeds")
```
The structure of the `DeepWeeds` folder is as follows:
```
└── root dir
    └── DeepWeeds //Contains folders of images for each class.
	├── rangeland_weeds_australia
		├── Classes
``` 

## <a name="CitingDeepWeeds"></a>Citing DeepWeeds
If you use this dataset in your research, please cite the following paper:
<a href="https://doi.org/10.1038/s41598-018-38343-3"> DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning </a>


**BibTex:**
```
@article{DeepWeeds2019,
  author = {Alex Olsen and
    Dmitry A. Konovalov and
    Bronson Philippa and
    Peter Ridd and
    Jake C. Wood and
    Jamie Johns and
    Wesley Banks and
    Benjamin Girgenti and
    Owen Kenny and 
    James Whinney and
    Brendan Calvert and
    Mostafa {Rahimi Azghadi} and
    Ronald D. White},
  title = {{DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning}},
  journal = {Scientific Reports},
  year = 2019,
  number = 2058,
  month = 2,
  volume = 9,
  issue = 1,
  day = 14,
  url = "https://doi.org/10.1038/s41598-018-38343-3",
  doi = "10.1038/s41598-018-38343-3"
}
```

## <a name="CitingAgML"></a>Citing AgML

**BibTex:**
```
@misc{joshi2021agml,
  author = {Amogh Joshi et al.},
  title = {AgML},
  year = {2021},
  month = {November 5},
  publisher = {GitHub},
  url = {https://github.com/Project-AgML/AgML}
}

```

