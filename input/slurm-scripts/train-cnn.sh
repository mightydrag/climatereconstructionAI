#!/bin/bash

#SBATCH -J crai-training
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --account=uo1075
#SBATCH --time=10:00:00
#SBATCH --mem=480G
#SBATCH --constraint a100_80
#SBATCH --mail-user=mats.militzer@studium.uni-hamburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/uo1075/u244085/trees_in_the_city/Physics-Informed-CNN-Reconstructions/input/slurm-scripts/train_slurm_output/train-cnn.o%j 
#SBATCH --error=/work/uo1075/u244085/trees_in_the_city/Physics-Informed-CNN-Reconstructions/input/slurm-scripts/train_slurm_output/train-cnn.e%j
source ~/.bashrc
#module load python3
mamba activate crai
cd /work/uo1075/u244085/trees_in_the_city/Physics-Informed-CNN-Reconstructions

python -m climatereconstructionai.train --load-from-file ./input/levante/train-crai-downscalling.txt --max-iter 4000
#python -m climatereconstructionai.train --load-from-file ./input/levante/train-crai-downscalling.txt --max-iter 250000 --resume-iter 125000
#python -m climatereconstructionai.train --load-from-file ./input/levante/train-crai-downscalling.txt --max-iter 1000000 --resume-latest

