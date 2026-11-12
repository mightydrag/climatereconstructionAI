#!/bin/bash
# slurm-eval-crai.sh — evaluation job; ALL run parameters are read from the
# config file passed via CONFIG_FILE (see submit-eval-cerra.sh).
#SBATCH -J crai-eval
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --account=kd1418
#SBATCH --time=00:15:00
#SBATCH --mem=480G
#SBATCH --constraint a100_80
#SBATCH --mail-user=mats.militzer@studium.uni-hamburg.de
#SBATCH --mail-type=BEGIN,END,FAIL,TIMEOUT
#SBATCH --output=eval_slurm_output/%x.o%j
#SBATCH --error=eval_slurm_output/%x.e%j

source ~/.bashrc || true
set -euo pipefail
mamba activate crai

REPO_DIR="${REPO_DIR:-/work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions}"
cd "$REPO_DIR"

CONFIG_FILE="${CONFIG_FILE:?set CONFIG_FILE}"
CFG="./input/levante/$CONFIG_FILE"
[ -f "$CFG" ] || { echo "config not found: $CFG"; exit 1; }

# get_cfg <key> [default] — value of "--<key> <value>" from the config;
# exits 1 if the key is missing and no default is given
get_cfg() {
    local v
    v=$(sed -n "s/^--$1[[:space:]]\+\(.*\)$/\1/p" "$CFG" | head -1 | tr -d '[:space:]')
    if [ -z "$v" ]; then
        if [ $# -ge 2 ]; then printf '%s' "$2"; return; fi
        echo "missing required --$1 in $CFG"; exit 1
    fi
    printf '%s' "$v"
}

type2var() {
    case "$1" in
        2t) echo t2m ;; orog) echo orography ;; lsm) echo land_sea_mask ;;
        tp) echo tp ;; 2r) echo relative_humidity ;; sp) echo surface_pressure ;;
        10si) echo wind_speed ;; 10wdir) echo wind_direction ;; *) return 1 ;;
    esac
}

DT=$(get_cfg data-types)
N_TARGET=$(get_cfg n-target-data 1)
REG=$(get_cfg region ger)
START_YEAR=$(get_cfg start-year)
END_YEAR=$(get_cfg end-year)
MODEL_NAMES=$(get_cfg model-names best)
STATIC_TYPES=$(get_cfg static-vars orog)

DATA_TPL=$(get_cfg data-file-template)
TARGET_TPL=$(get_cfg target-file-template)
MASK_TPL=$(get_cfg mask-file-template)
STEADY_TPL=$(get_cfg steady-mask-file)

IFS=',' read -ra TYPES <<< "$DT"
VARS=()
for t in "${TYPES[@]}"; do
    v=$(type2var "$t") || { echo "unknown data type '$t' in $CFG"; exit 1; }
    VARS+=("$v")
done
n=${#VARS[@]}
TARGET_VAR="${VARS[$((n-N_TARGET))]}"
INPUT_VARS=$(printf '%s ' "${VARS[@]:0:$((n-N_TARGET))}")

STATIC_VARS=""
IFS=',' read -ra S_TYPES <<< "$STATIC_TYPES"
for t in "${S_TYPES[@]}"; do
    v=$(type2var "$t") || { echo "unknown static data type '$t' in $CFG"; exit 1; }
    STATIC_VARS+="$v "
done
if [ -n "$STATIC_VARS" ]; then
    STATIC_TPL=$(get_cfg static-file-template)
    STATIC_MASK_TPL=$(get_cfg static-mask-template)
else
    STATIC_TPL=""; STATIC_MASK_TPL=""
fi

subst() { echo "$1" | sed "s/{year}/$2/g; s/{var}/$3/g; s/{reg}/$REG/g"; }

echo "================ slurm job ==================="
echo "job id      : ${SLURM_JOB_ID:-none}"
echo "node(s)     : ${SLURM_NODELIST:-localhost}"
echo "config      : $CFG"
echo "years       : $START_YEAR..$END_YEAR"
echo "data-types  : $DT"
echo "INPUT_VARS  : $INPUT_VARS"
echo "TARGET_VAR  : $TARGET_VAR"
echo "STATIC_VARS : $STATIC_VARS"
echo "REG         : $REG"
echo "model-names : $MODEL_NAMES"
echo "=============================================="

processed=0; failed=0; skipped=0
for ((year=START_YEAR; year<=END_YEAR; year++)); do
    DATA_NAMES=""; skip=0
    for v in $INPUT_VARS; do
        if [[ " $STATIC_VARS " == *" $v "* ]]; then f=$(subst "$STATIC_TPL" "" "$v")
        else f=$(subst "$DATA_TPL" "$year" "$v"); fi
        if [ ! -f "$f" ]; then echo "=== SKIP $year: missing $f ==="; skip=1; break; fi
        DATA_NAMES+="${f},"
    done
    TARGET_FILE=$(subst "$TARGET_TPL" "$year" "$TARGET_VAR")
    if [ $skip -eq 0 ] && [ ! -f "$TARGET_FILE" ]; then
        echo "=== SKIP $year: missing $TARGET_FILE ==="; skip=1
    fi
    [ $skip -eq 1 ] && { skipped=$((skipped+1)); continue; }
    DATA_NAMES+="${TARGET_FILE}"

    MASK_NAMES=""
    for v in $INPUT_VARS $TARGET_VAR; do
        if [[ " $STATIC_VARS " == *" $v "* ]]; then f=$(subst "$STATIC_MASK_TPL" "" "$v")
        else f=$(subst "$MASK_TPL" "$year" "$v"); fi
        if [ ! -f "$f" ]; then echo "=== SKIP $year: missing mask $f ==="; skip=1; break; fi
        MASK_NAMES+="${f},"
    done
    [ $skip -eq 1 ] && { skipped=$((skipped+1)); continue; }
    MASK_NAMES="${MASK_NAMES%,}"
    STEADY_MASK=$(subst "$STEADY_TPL" "" "$TARGET_VAR")

    N_TYPES=$(echo "$DT" | awk -F',' '{print NF}')
    N_NAMES=$(echo "$DATA_NAMES" | awk -F',' '{print NF}')
    if [ "$N_TYPES" -ne "$N_NAMES" ]; then
        echo "=== SKIP $year: $N_NAMES data names vs $N_TYPES data types in $CFG ==="
        skipped=$((skipped+1)); continue
    fi

    echo "=== Processing year $year ==="
    if python3 -m climatereconstructionai.evaluate \
        --load-from-file "$CFG" \
        --data-names "$DATA_NAMES" \
        --mask-names "$MASK_NAMES" \
        --steady-masks "$STEADY_MASK" \
        --xrdss-paths "$(echo "$DATA_NAMES" | cut -d',' -f1)" \
        --eval-names "cerra_dm_${TARGET_VAR}_${REG}_${year}_crai" \
        --model-names "$MODEL_NAMES"
    then
        echo "=== Completed year $year ==="
        processed=$((processed+1))
    else
        echo "=== FAILED year $year (exit $?) ==="
        failed=$((failed+1))
    fi
done

echo "=============================================="
echo "Done: $processed evaluated, $failed failed, $skipped skipped (years $START_YEAR..$END_YEAR)"
[ "$failed" -eq 0 ] || exit 1
