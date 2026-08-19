#!/bin/bash
# This script submits a chain of dependent jobs for long training

# ------------------------------
# 1️⃣ Parameters
# ------------------------------
TRAIN_SCRIPT="/work/uo1075/u241308/ML_downscaling/code/climatereconstructionAI/input/slurm-scripts/train-cnn.sh"
NUM_JOBS=10   # number of 10h jobs to pre-submit

# ------------------------------
# 2️⃣ Submit the first job
# ------------------------------
jid_prev=$(sbatch "$TRAIN_SCRIPT" | awk '{print $4}')
echo "Submitted job 1: $jid_prev"

# ------------------------------
# 3️⃣ Submit dependent jobs
# ------------------------------
for i in $(seq 2 $NUM_JOBS); do
    jid_prev=$(sbatch --dependency=afterany:$jid_prev "$TRAIN_SCRIPT" | awk '{print $4}')
    echo "Submitted job $i with dependency on previous: $jid_prev"
done

echo "All $NUM_JOBS jobs submitted with proper dependencies."
