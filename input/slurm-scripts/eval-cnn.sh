#!/bin/bash

#SBATCH -J crai-eval
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --account=uo1075
#SBATCH --time=10:00:00
#SBATCH --mem=480G
#SBATCH --constraint a100_80
#SBATCH --mail-user=mats.militzer@studium.uni-hamburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/uo1075/u244085/trees_in_the_city/Physics-Informed-CNN-Reconstructions/input/slurm-scripts/eval_slurm_output/eval-cnn.o%j 
#SBATCH --error=/work/uo1075/u244085/trees_in_the_city/Physics-Informed-CNN-Reconstructions/input/slurm-scripts/eval_slurm_output/eval-cnn.e%j

source ~/.bashrc
#module load python3
mamba activate crai
#module load python3/2022.01-gcc-11.2.0
cd /work/uo1075/u244085/trees_in_the_city/Physics-Informed-CNN-Reconstructions/

#source activate crai
python -m climatereconstructionai.evaluate --load-from-file ./input/levante/eval-crai-downscalling.txt
