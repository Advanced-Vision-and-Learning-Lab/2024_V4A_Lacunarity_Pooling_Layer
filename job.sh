#!/bin/bash
#SBATCH --job-name=Lacunarity	                # Job name
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=akshatha.mohan@tamu.edu     # Where to send mail
#SBATCH --ntasks=12                             # Run on 12 CPU cores (max for Intel Core i9-10920X CPU)
#SBATCH --partition=gpu                         # Select GPU Partition
#SBATCH --time=48:00:00                         # Time limit hrs:min:sec 36 hour max
#SBATCH --nodes=1
#SBATCH --output=/home/grads/a/akshatha.mohan/Documents/Thesis/Lacunarity_Pooling/logs/%j_log.out

EPOCH=200
DATASET1=2
DATASET2=3
scales0to1="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
scales1to2="1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0"
scales0to2="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0"
scales0to3="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0"
scales1to3="1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0"
scales0to5="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.0"
scales1to5="1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.0"


cd /home/grads/a/akshatha.mohan/Documents/Thesis/Lacunarity_Pooling

conda activate base

cd /home/grads/a/akshatha.mohan/Documents/Thesis/Lacunarity_Pooling

#Dataset = 1:PneumoniaMNIST, 2:BloodMNIST, 3:OrganMNISTCoronal, 4:FashionMNIST'
#Pooling layer = 1:max, 2:avg, 3:Pixel_Lacunarity, 4:ScalePyramid, 5:BuildPyramid'

##########BASELINE#####################################################

#-------------MAX-------------------------#

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 1 --kernel 2 --stride 1 --folder 'Saved_Models/k=2'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 1 --kernel 2 --stride 1 --folder 'Saved_Models/k=2'

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 1 --kernel 4 --stride 1 --folder 'Saved_Models/k=4'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 1 --kernel 4 --stride 1 --folder 'Saved_Models/k=4'

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 1 --kernel 3 --stride 1 --folder 'Saved_Models/k=3'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 1 --kernel 3 --stride 1 --folder 'Saved_Models/k=3'

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 1 --kernel 5 --stride 1 --folder 'Saved_Models/k=5'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 1 --kernel 5 --stride 1 --folder 'Saved_Models/k=5'


#-------------AVG-------------------------#

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 2 --kernel 2 --stride 1 --folder 'Saved_Models/k=2'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 2 --kernel 2 --stride 1 --folder 'Saved_Models/k=2'

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 2 --kernel 4 --stride 1 --folder 'Saved_Models/k=4'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 2 --kernel 4 --stride 1 --folder 'Saved_Models/k=4'

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 2 --kernel 3 --stride 1 --folder 'Saved_Models/k=3'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 2 --kernel 3 --stride 1 --folder 'Saved_Models/k=3'

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 2 --kernel 5 --stride 1 --folder 'Saved_Models/k=5'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 2 --kernel 5 --stride 1 --folder 'Saved_Models/k=5'



##########BLOODMNIST####################################################
# Run your Python script

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 2 --stride 1 --scales $scales0to1 --folder 'Saved_Models/pixellacunarity/k=2_s=1_0to1range'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 2 --stride 1 --scales $scales0to1 --folder 'Saved_Models/pixellacunarity/k=2_s=1_0to1range'

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 2 --stride 1 --scales $scales1to2 --folder 'Saved_Models/pixellacunarity/k=2_s=1_1to2range'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 2 --stride 1 --scales $scales1to2 --folder 'Saved_Models/pixellacunarity/k=2_s=1_1to2range'

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 2 --stride 1 --scales $scales0to2 --folder 'Saved_Models/pixellacunarity/k=2_s=1_0to2range'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 2 --stride 1 --scales $scales0to2 --folder 'Saved_Models/pixellacunarity/k=2_s=1_0to2range'

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 2 --stride 1 --scales $scales0to3 --folder 'Saved_Models/pixellacunarity/k=2_s=1_0to3range'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 2 --stride 1 --scales $scales0to3 --folder 'Saved_Models/pixellacunarity/k=2_s=1_0to3range'

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 2 --stride 1 --scales $scales1to3 --folder 'Saved_Models/pixellacunarity/k=2_s=1_1to3range'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 2 --stride 1 --scales $scales1to3 --folder 'Saved_Models/pixellacunarity/k=2_s=1_1to3range'

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 2 --stride 1 --scales $scales0to5 --folder 'Saved_Models/pixellacunarity/k=2_s=1_0to5range'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 2 --stride 1 --scales $scales0to5 --folder 'Saved_Models/pixellacunarity/k=2_s=1_0to5range'

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 2 --stride 1 --scales $scales1to5 --folder 'Saved_Models/pixellacunarity/k=2_s=1_1to5range'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 2 --stride 1 --scales $scales1to5 --folder 'Saved_Models/pixellacunarity/k=2_s=1_1to5range'




# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 4 --stride 1 --scales $scales0to1 --folder 'Saved_Models/pixellacunarity/k=4_s=1_0to1range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 4 --stride 1 --scales $scales0to1 --folder 'Saved_Models/pixellacunarity/k=4_s=1_0to1range'

# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 4 --stride 1 --scales $scales1to2 --folder 'Saved_Models/pixellacunarity/k=4_s=1_1to2range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 4 --stride 1 --scales $scales1to2 --folder 'Saved_Models/pixellacunarity/k=4_s=1_1to2range'

# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 4 --stride 1 --scales $scales0to2 --folder 'Saved_Models/pixellacunarity/k=4_s=1_0to2range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 4 --stride 1 --scales $scales0to2 --folder 'Saved_Models/pixellacunarity/k=4_s=1_0to2range'

# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 4 --stride 1 --scales $scales0to3 --folder 'Saved_Models/pixellacunarity/k=4_s=1_0to3range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 4 --stride 1 --scales $scales0to3 --folder 'Saved_Models/pixellacunarity/k=4_s=1_0to3range'

# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 4 --stride 1 --scales $scales1to3 --folder 'Saved_Models/pixellacunarity/k=4_s=1_1to3range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 4 --stride 1 --scales $scales1to3 --folder 'Saved_Models/pixellacunarity/k=4_s=1_1to3range'

# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 4 --stride 1 --scales $scales0to5 --folder 'Saved_Models/pixellacunarity/k=4_s=1_0to5range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 4 --stride 1 --scales $scales0to5 --folder 'Saved_Models/pixellacunarity/k=4_s=1_0to5range'

# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 4 --stride 1 --scales $scales1to5 --folder 'Saved_Models/pixellacunarity/k=4_s=1_1to5range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 4 --stride 1 --scales $scales1to5 --folder 'Saved_Models/pixellacunarity/k=4_s=1_1to5range'




# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 5 --stride 1 --scales $scales0to1 --folder 'Saved_Models/pixellacunarity/k=5_s=1_0to1range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 5 --stride 1 --scales $scales0to1 --folder 'Saved_Models/pixellacunarity/k=5_s=1_0to1range'

# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 5 --stride 1 --scales $scales1to2 --folder 'Saved_Models/pixellacunarity/k=5_s=1_1to2range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 5 --stride 1 --scales $scales1to2 --folder 'Saved_Models/pixellacunarity/k=5_s=1_1to2range'

# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 5 --stride 1 --scales $scales0to2 --folder 'Saved_Models/pixellacunarity/k=5_s=1_0to2range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 5 --stride 1 --scales $scales0to2 --folder 'Saved_Models/pixellacunarity/k=5_s=1_0to2range'

# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 5 --stride 1 --scales $scales0to3 --folder 'Saved_Models/pixellacunarity/k=5_s=1_0to3range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 5 --stride 1 --scales $scales0to3 --folder 'Saved_Models/pixellacunarity/k=5_s=1_0to3range'

# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 5 --stride 1 --scales $scales1to3 --folder 'Saved_Models/pixellacunarity/k=5_s=1_1to3range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 5 --stride 1 --scales $scales1to3 --folder 'Saved_Models/pixellacunarity/k=5_s=1_1to3range'

# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 5 --stride 1 --scales $scales0to5 --folder 'Saved_Models/pixellacunarity/k=5_s=1_0to5range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 5 --stride 1 --scales $scales0to5 --folder 'Saved_Models/pixellacunarity/k=5_s=1_0to5range'

# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 5 --stride 1 --scales $scales1to5 --folder 'Saved_Models/pixellacunarity/k=5_s=1_1to5range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 5 --stride 1 --scales $scales1to5 --folder 'Saved_Models/pixellacunarity/k=5_s=1_1to5range'






# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 3 --stride 1 --scales $scales0to1 --folder 'Saved_Models/pixellacunarity/k=3_s=1_0to1range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 3 --stride 1 --scales $scales0to1 --folder 'Saved_Models/pixellacunarity/k=3_s=1_0to1range'

# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 3 --stride 1 --scales $scales1to2 --folder 'Saved_Models/pixellacunarity/k=3_s=1_1to2range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 3 --stride 1 --scales $scales1to2 --folder 'Saved_Models/pixellacunarity/k=3_s=1_1to2range'

# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 3 --stride 1 --scales $scales0to2 --folder 'Saved_Models/pixellacunarity/k=3_s=1_0to2range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 3 --stride 1 --scales $scales0to2 --folder 'Saved_Models/pixellacunarity/k=3_s=1_0to2range'

# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 3 --stride 1 --scales $scales0to3 --folder 'Saved_Models/pixellacunarity/k=3_s=1_0to3range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 3 --stride 1 --scales $scales0to3 --folder 'Saved_Models/pixellacunarity/k=3_s=1_0to3range'

# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 3 --stride 1 --scales $scales1to3 --folder 'Saved_Models/pixellacunarity/k=3_s=1_1to3range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 3 --stride 1 --scales $scales1to3 --folder 'Saved_Models/pixellacunarity/k=3_s=1_1to3range'

# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 3 --stride 1 --scales $scales0to5 --folder 'Saved_Models/pixellacunarity/k=3_s=1_0to5range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 3 --stride 1 --scales $scales0to5 --folder 'Saved_Models/pixellacunarity/k=3_s=1_0to5range'

# python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 3 --stride 1 --scales $scales1to5 --folder 'Saved_Models/pixellacunarity/k=3_s=1_1to5range'
# python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 3 --kernel 3 --stride 1 --scales $scales1to5 --folder 'Saved_Models/pixellacunarity/k=3_s=1_1to5range'



#####BUILD PYRAMID############################################################################################

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 5 --kernel 2 --stride 1 --num_levels 3  --folder 'Saved_Models/Build_Pyramid/k=2_s=1_3_levels'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 5 --kernel 2 --stride 1 --num_levels 3  --folder 'Saved_Models/Build_Pyramid/k=2_s=1_3_levels'

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 5 --kernel 3 --stride 1 --num_levels 3  --folder 'Saved_Models/Build_Pyramid/k=3_s=1_3_levels'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 5 --kernel 3 --stride 1 --num_levels 3  --folder 'Saved_Models/Build_Pyramid/k=3_s=1_3_levels'

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 5 --kernel 4 --stride 1 --num_levels 2  --folder 'Saved_Models/Build_Pyramid/k=4_s=1_2_levels'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 5 --kernel 4 --stride 1 --num_levels 2  --folder 'Saved_Models/Build_Pyramid/k=4_s=1_2_levels'

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 5 --kernel 5 --stride 1 --num_levels 2  --folder 'Saved_Models/Build_Pyramid/k=5_s=1_2_levels'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 5 --kernel 5 --stride 1 --num_levels 2  --folder 'Saved_Models/Build_Pyramid/k=5_s=1_2_levels'

python demo.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 5 --kernel 6 --stride 1 --num_levels 2  --folder 'Saved_Models/Build_Pyramid/k=6_s=1_2_levels'
python View_Results.py --data_selection $DATASET1 --num_epochs $EPOCH --pooling_layer 5 --kernel 6 --stride 1 --num_levels 2  --folder 'Saved_Models/Build_Pyramid/k=6_s=1_2_levels'


python demo.py --data_selection 2 --num_epochs 200 --pooling_layer 3 --kernel 5 --stride 1 --scales $scales1to5 --folder 'Saved_Models/pixellacunarity/k=5_s=1_1to5range'