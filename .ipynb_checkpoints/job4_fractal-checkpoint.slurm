#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=lacunarity          #Set the job name to "JobExample4"
#SBATCH --time=10:00:00              #Set the time. 
#SBATCH --nodes=2                    #Request 2 nodes
#SBATCH --ntasks-per-node=32         #Request 32 tasks/cores per node
#SBATCH --mem=128G                   #Request 248G (248GB) per node
#SBATCH --output=lacunarity.%j             #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:a100:2                      #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu                         #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456                       #Set billing account to 123456
##SBATCH --mail-type=ALL                        #Send email on all job events
##SBATCH --mail-user=akshatha.mohan@tamu.edu    #Send all emails to email_address

ml purge

#END_OF_INSERTION

#First Executable Line
module purge
cd /scratch/user/akshatha.mohan/research/Lacunarity_Pooling;
module load Anaconda3/2022.10;
source activate myenv;


EPOCH=100
DATASET1=13
DATASET2=14
DATASET3=12


#Dataset_names = {1:'PneumoniaMNIST', 2:'BloodMNIST',3:'OrganMNISTCoronal', 4: 'FashionMNIST',5: 'PlantLeaf', 6: 'UCMerced', 7: 'PRMI', 8: 'Synthetic_Gray', 9: 'Synthetic_RGB',
             #        10: 'Kth_Tips', 11: 'GTOS-mobile', 12: 'LeavesTex', 13: 'PlantVillage', 14: 'DeepWeeds', 15: 'Cassava'}
#1:max, 2:avg, 3:Base_Lacunarity, 4:Pixel_Lacunarity, 5:ScalePyramid_Lacunarity, 6:BuildPyramid, 7:DBC, 8:GDCB, 9: Baseline
##########BASELINE#####################################################
#-------------DATASET1------------------------#

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 9 --fractal --model resnet18_lacunarity --folder 'Saved_Models/fractal'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 9 --fractal --model resnet18_lacunarity --folder 'Saved_Models/fractal'

# python demo.py --data_selection $DATASET2 --num_epochs $EPOCH --pooling_layer 9 --fractal --model resnet18_lacunarity --folder 'Saved_Models/fractal'
# python View_Results.py --data_selection $DATASET2 --num_epochs $EPOCH --pooling_layer 9 --fractal --model resnet18_lacunarity --folder 'Saved_Models/fractal'

# python demo.py --data_selection $DATASET3 --num_epochs $EPOCH --pooling_layer 9 --fractal --model resnet18_lacunarity --folder 'Saved_Models/fractal'
# python View_Results.py --data_selection $DATASET3 --num_epochs $EPOCH --pooling_layer 9 --fractal --model resnet18_lacunarity --folder 'Saved_Models/fractal'




# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 9 --fractal --model convnext_lacunarity --folder 'Saved_Models/fractal'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 9 --fractal --model convnext_lacunarity --folder 'Saved_Models/fractal'

# python demo.py --data_selection $DATASET2 --num_epochs $EPOCH --pooling_layer 9 --fractal --model convnext_lacunarity --folder 'Saved_Models/fractal'
# python View_Results.py --data_selection $DATASET2 --num_epochs $EPOCH --pooling_layer 9 --fractal --model convnext_lacunarity --folder 'Saved_Models/fractal'

# python demo.py --data_selection $DATASET3 --num_epochs $EPOCH --pooling_layer 9 --fractal --model convnext_lacunarity --folder 'Saved_Models/fractal'
# python View_Results.py --data_selection $DATASET3 --num_epochs $EPOCH --pooling_layer 9 --fractal --model convnext_lacunarity --folder 'Saved_Models/fractal'





python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 9 --fractal --model densenet161_lacunarity --folder 'Saved_Models/fractal'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 9 --fractal --model densenet161_lacunarity --folder 'Saved_Models/fractal'

python demo.py --data_selection $DATASET2 --num_epochs $EPOCH --pooling_layer 9 --fractal --model densenet161_lacunarity --folder 'Saved_Models/fractal'
python View_Results.py --data_selection $DATASET2 --num_epochs $EPOCH --pooling_layer 9 --fractal --model densenet161_lacunarity --folder 'Saved_Models/fractal'

python demo.py --data_selection $DATASET3 --num_epochs $EPOCH --pooling_layer 9 --fractal --model densenet161_lacunarity --folder 'Saved_Models/fractal'
python View_Results.py --data_selection $DATASET3 --num_epochs $EPOCH --pooling_layer 9 --fractal --model densenet161_lacunarity --folder 'Saved_Models/fractal'




python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 9 --fractal --model efficientnet_lacunarity --folder 'Saved_Models/fractal'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 9 --fractal --model efficientnet_lacunarity --folder 'Saved_Models/fractal'

python demo.py --data_selection $DATASET2 --num_epochs $EPOCH --pooling_layer 9 --fractal --model efficientnet_lacunarity --folder 'Saved_Models/fractal'
python View_Results.py --data_selection $DATASET2 --num_epochs $EPOCH --pooling_layer 9 --fractal --model efficientnet_lacunarity --folder 'Saved_Models/fractal'

python demo.py --data_selection $DATASET3 --num_epochs $EPOCH --pooling_layer 9 --fractal --model efficientnet_lacunarity --folder 'Saved_Models/fractal'
python View_Results.py --data_selection $DATASET3 --num_epochs $EPOCH --pooling_layer 9 --fractal --model efficientnet_lacunarity --folder 'Saved_Models/fractal'














