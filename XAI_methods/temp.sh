#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=lacunarity          #Set the job name to "JobExample4"
#SBATCH --time=16:00:00              #Set the time. 
#SBATCH --ntasks=8                   #Request 1 task
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

EPOCH=50
EPOCH2=400
DATASET1=10
DATASET2=11
DATASET3=12
scales0to1="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
scales1to2="1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0"
scales0to2="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0"
scales0to3="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0"
scales1to3="1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0"
scales0to5="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.0"
scales1to5="1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.0"
sigma1="0.1"
sigma2="0.2"
sigma3="0.4"
sigma4="0.6"

#pooling layer = 1:max, 2:avg, 3:Base_Lacunarity, 4:Pixel_Lacunarity, 5:ScalePyramid_Lacunarity, 6:BuildPyramid, 7:DBC, 8:GDCB, 9: Baseline'
#dataset =1:PneumoniaMNIST, 2:BloodMNIST, 3:OrganMNISTCoronal, 4:FashionMNIST, 5:PlantLeaf, 6:UCMerced, 7:PRMI, 8:Synthetic_Gray, 9:Synthetic_RGB, 10:Kth_Tips, 11: GTOS-mobile, 12:LeavesTex

################BASELINE###############################################

python demo.py --data_selection $DATASET1 --model convnext --agg_func 1 --num_epochs $EPOCH2 --pooling_layer 9  --folder 'Saved_Models/Baseline_global/'
python View_Results.py --data_selection $DATASET1 --model convnext --agg_func 1 --num_epochs $EPOCH2 --pooling_layer 9  --folder 'Saved_Models/Baseline_global/'

python demo.py --data_selection $DATASET2 --model convnext --agg_func 1 --num_epochs $EPOCH2 --pooling_layer 9  --folder 'Saved_Models/Baseline_global/'
python View_Results.py --data_selection $DATASET2 --model convnext --agg_func 1 --num_epochs $EPOCH2 --pooling_layer 9  --folder 'Saved_Models/Baseline_global/'

python demo.py --data_selection $DATASET3 --model convnext --agg_func 1 --num_epochs $EPOCH2 --pooling_layer 9  --folder 'Saved_Models/Baseline_global/'
python View_Results.py --data_selection $DATASET3 --model convnext --agg_func 1 --num_epochs $EPOCH2 --pooling_layer 9  --folder 'Saved_Models/Baseline_global/'


################Lacunarity layers########################################


python demo.py --data_selection $DATASET1 --model convnext --num_epochs $EPOCH2 --agg_func 1 --pooling_layer 4  --folder 'Saved_Models/Learnable_Lacunarity_Global'
python View_Results.py --data_selection $DATASET1 --model convnext --num_epochs $EPOCH2 --agg_func 1 --pooling_layer 4  --folder 'Saved_Models/Learnable_Lacunarity_Global'

python demo.py --data_selection $DATASET2 --model convnext --num_epochs $EPOCH2 --agg_func 1 --pooling_layer 4  --folder 'Saved_Models/Learnable_Lacunarity_Global'
python View_Results.py --data_selection $DATASET2 --model convnext --num_epochs $EPOCH2 --agg_func 1 --pooling_layer 4  --folder 'Saved_Models/Learnable_Lacunarity_Global'

python demo.py --data_selection $DATASET3 --model convnext --num_epochs $EPOCH2 --agg_func 1 --pooling_layer 4  --folder 'Saved_Models/Learnable_Lacunarity_Global'
python View_Results.py --data_selection $DATASET3 --model convnext --num_epochs $EPOCH2 --agg_func 1 --pooling_layer 4  --folder 'Saved_Models/Learnable_Lacunarity_Global'


python demo.py --data_selection $DATASET1 --model convnext --num_epochs $EPOCH2 --agg_func 2 --kernel 4 --stride 1 --pooling_layer 4  --folder 'Saved_Models/Learnable_Lacunarity_local'
python View_Results.py --data_selection $DATASET1 --model convnext --num_epochs $EPOCH2 --agg_func 1 --kernel 4 --stride 1 --pooling_layer 4  --folder 'Saved_Models/Learnable_Lacunarity_local'

python demo.py --data_selection $DATASET2 --model convnext --num_epochs $EPOCH2 --agg_func 2 --kernel 4 --stride 1 --pooling_layer 4  --folder 'Saved_Models/Learnable_Lacunarity_local'
python View_Results.py --data_selection $DATASET2 --model convnext --num_epochs $EPOCH2 --agg_func 2 --kernel 4 --stride 1 --pooling_layer 4  --folder 'Saved_Models/Learnable_Lacunarity_local'

python demo.py --data_selection $DATASET3 --model convnext --num_epochs $EPOCH2 --agg_func 2 --kernel 4 --stride 1 --pooling_layer 4  --folder 'Saved_Models/Learnable_Lacunarity_local'
python View_Results.py --data_selection $DATASET3 --model convnext --num_epochs $EPOCH2 --agg_func 2 --kernel 4 --stride 1 --pooling_layer 4  --folder 'Saved_Models/Learnable_Lacunarity_local'
