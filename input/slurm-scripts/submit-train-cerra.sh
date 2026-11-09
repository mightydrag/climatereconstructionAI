#!/bin/bash
# Submit wrapper: derives job name and log file names from the config file name.
#
# Usage:
#   ./submit_train.sh                                  # uses default config
#   ./submit_train.sh train-cnn-cerra-t2m-ger.txt      # tag = t2m-ger
#
# For config "train-cnn-cerra-t2m-ger.txt" this submits:
#   job name : crai-training-t2m-ger
#   stdout   : .../train-cnn-cerra-t2m-ger.o<jobid>
#   stderr   : .../train-cnn-cerra-t2m-ger.e<jobid>

set -euo pipefail

# --- settings --------------------------------------------------------------
PREFIX="train-cnn-cerra-"
JOB_SCRIPT="slurm-train-crai.sh"   # the sbatch file with the SBATCH resource headers
LOG_DIR="/work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions/input/slurm-scripts/train_slurm_output"
JOB_NAME_PREFIX="crai-training"
DEFAULT_CONFIG="train-cnn-cerra-t2m.txt"

# --- derive config file and tag ---------------------------------------------
CONFIG_FILE="${1:-$DEFAULT_CONFIG}"

# strip directory part if a path was given
BASENAME=$(basename "$CONFIG_FILE")

# strip extension (.txt or anything else)
TAG="${BASENAME%.*}"

# strip the leading "train-cnn-cerra-" prefix
TAG="${TAG#$PREFIX}"

# fallback if the prefix was not present or nothing is left
if [[ -z "$TAG" ]]; then
    TAG="default"
fi

CONFIG_FILE="$BASENAME"

echo "Submitting job for config: $CONFIG_FILE"
echo "  job name : ${JOB_NAME_PREFIX}-${TAG}"
echo "  logs     : ${LOG_DIR}/train-cnn-cerra-${TAG}.o<jobid> / .e<jobid>"

# --- submit -------------------------------------------------------------------
# Job name and log names cannot be set from $1 inside #SBATCH lines (those are
# read at submission time, before the shell runs), so they are passed here.
sbatch \
    --job-name="${JOB_NAME_PREFIX}-${TAG}" \
    --output="${LOG_DIR}/train-cnn-cerra-${TAG}.o%j" \
    --error="${LOG_DIR}/train-cnn-cerra-${TAG}.e%j" \
    "$JOB_SCRIPT" "$CONFIG_FILE"