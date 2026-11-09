#!/bin/bash
# submit_eval.sh — submit ONE evaluation job; the year loop runs inside the slurm job
# usage: ./submit_eval.sh [config_file]
REPO_DIR="/work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions"
JOB_SCRIPT="${JOB_SCRIPT:-$REPO_DIR/input/slurm-scripts/slurm-eval-crai.sh}"
CONFIG_FILE="${1:-eval-cnn-cerra-relhum-ger.txt}"
CONFIG_PATH="$REPO_DIR/input/levante/$CONFIG_FILE"
[ -f "$CONFIG_PATH" ] || { echo "config not found: $CONFIG_PATH"; exit 1; }

get_cfg() { sed -n "s/^--$1[[:space:]]\\+\\(.*\\)$/\\1/p" "$CONFIG_PATH" | head -1 | tr -d '[:space:]'; }

type2var() {
    case "$1" in
        2t) echo t2m ;; orog) echo orography ;; lsm) echo land_sea_mask ;;
        tp) echo tp ;; 2r) echo relative_humidity ;; sp) echo surface_pressure ;;
        10si) echo wind_speed ;; 10wdir) echo wind_direction ;; *) return 1 ;;
    esac
}

DT=$(get_cfg data-types)
N_TARGET=$(get_cfg n-target-data); N_TARGET="${N_TARGET:-1}"
REG=$(get_cfg region); REG="${REG:-ger}"

IFS=',' read -ra TYPES <<< "$DT"
VARS=()
for t in "${TYPES[@]}"; do
    v=$(type2var "$t") || { echo "unknown data type '$t' in $CONFIG_FILE"; exit 1; }
    VARS+=("$v")
done
n=${#VARS[@]}
TARGET_VAR="${VARS[$((n-N_TARGET))]}"
INPUT_VARS=$(printf '%s ' "${VARS[@]:0:$((n-N_TARGET))}")
STATIC_VARS="${STATIC_VARS:-orography}"

JOB_NAME="crai-eval-${CONFIG_FILE%.txt}"   # slurm job name from the config file

echo "================ submit_eval ================"
echo "config      : $CONFIG_PATH"
echo "job name    : $JOB_NAME"
echo "data-types  : $DT"
echo "INPUT_VARS  : $INPUT_VARS"
echo "TARGET_VAR  : $TARGET_VAR"
echo "REG         : $REG"
echo "---- config content ----"
cat "$CONFIG_PATH"
echo "---- end config ----"
echo "=============================================="

sbatch -J "$JOB_NAME" \
    --output="eval_slurm_output/%x.o%j" \
    --error="eval_slurm_output/%x.e%j" \
    --export=ALL,CONFIG_FILE="$CONFIG_FILE",INPUT_VARS="$INPUT_VARS",TARGET_VAR="$TARGET_VAR",STATIC_VARS="$STATIC_VARS",REG="$REG" \
    "$JOB_SCRIPT"