#!/bin/bash

#SBATCH -J crai-training-t2m-ger
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --account=uo1075
#SBATCH --time=12:00:00
#SBATCH --mem=480G
#SBATCH --constraint a100_80
#SBATCH --mail-user=mats.militzer@studium.uni-hamburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions/input/slurm-scripts/train_slurm_output/train-cnn-cerra-t2m-ger.o%j 
#SBATCH --error=/work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions/input/slurm-scripts/train_slurm_output/train-cnn-cerra-t2m-ger.e%j
source ~/.bashrc
#module load python3
mamba activate crai
cd /work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions

#python3 -m climatereconstructionai.train --load-from-file ./input/levante/train-cnn-t2max-anomaly.txt --max-iter 250000
python3 -m climatereconstructionai.train --load-from-file ./input/levante/train-cnn-cerra-t2m-ger_sort_en4_po1.txt --max-iter 500000 #--resume-iter 70000
#python -m climatereconstructionai.train --load-from-file ./input/levante/train-crai-downscalling.txt --max-iter 1000000 --resume-latest

