#!/bin/bash

#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --account=kd1418
#SBATCH --time=06:00:00
#SBATCH --mem=480G
#SBATCH --constraint a100_80
#SBATCH --mail-user=mats.militzer@studium.uni-hamburg.de
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/.bashrc
#module load python3
mamba activate crai
cd /work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions

CONFIG_FILE="${1:-train-cnn-cerra-t2m.txt}" 


echo '---Printing Config---'
cat ./input/levante/$CONFIG_FILE
echo '--- Config END---'
#python3 -m climatereconstructionai.train --load-from-file ./input/levante/train-cnn-t2max-anomaly.txt --max-iter 250000
python3 -m climatereconstructionai.train --load-from-file ./input/levante/$CONFIG_FILE --max-iter 500000 #--resume-iter 70000
#python -m climatereconstructionai.train --load-from-file ./input/levante/train-crai-downscalling.txt --max-iter 1000000 --resume-latest

