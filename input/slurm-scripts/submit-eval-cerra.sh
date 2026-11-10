#!/bin/bash
# submit-eval-cerra.sh — submit ONE evaluation job; all run parameters are read
# from the config file inside the slurm job (slurm-eval-crai.sh).
# usage: ./submit-eval-cerra.sh [config_file]
set -euo pipefail

REPO_DIR="${REPO_DIR:-/work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions}"
JOB_SCRIPT="${JOB_SCRIPT:-$REPO_DIR/input/slurm-scripts/slurm-eval-crai.sh}"
CONFIG_FILE="${1:-eval-cnn-cerra-t2m-ger-el5-single-l0.txt}"
CONFIG_PATH="$REPO_DIR/input/levante/$CONFIG_FILE"
[ -f "$CONFIG_PATH" ] || { echo "config not found: $CONFIG_PATH"; exit 1; }

LOG_DIR="eval_slurm_output"
mkdir -p "$LOG_DIR"
JOB_NAME="crai-${CONFIG_FILE%.txt}"

echo "================ submit_eval ================"
echo "config      : $CONFIG_PATH"
echo "job name    : $JOB_NAME"
echo "logs        : $LOG_DIR/${JOB_NAME}.o<jobid> / .e<jobid>"
echo "=============================================="

sbatch -J "$JOB_NAME" \
    --output="$LOG_DIR/%x.o%j" \
    --error="$LOG_DIR/%x.e%j" \
    --export=ALL,CONFIG_FILE="$CONFIG_FILE" \
    "$JOB_SCRIPT"
