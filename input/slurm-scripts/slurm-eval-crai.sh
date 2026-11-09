#!/bin/bash
#SBATCH -J crai-eval
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --account=kd1418
#SBATCH --time=00:30:00
#SBATCH --mem=480G
#SBATCH --constraint a100_80
#SBATCH --mail-user=mats.militzer@studium.uni-hamburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=eval_slurm_output/eval-crai.o%j
#SBATCH --error=eval_slurm_output/eval-crai.e%j

source ~/.bashrc
mamba activate crai
cd /work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions/

CONFIG_FILE="${CONFIG_FILE:?set CONFIG_FILE}"
CFG="./input/levante/$CONFIG_FILE"
[ -f "$CFG" ] || { echo "config not found: $CFG"; exit 1; }

get_cfg() { sed -n "s/^--$1[[:space:]]\\+\\(.*\\)$/\\1/p" "$CFG" | head -1; }

type2var() {
    case "$1" in
        2t) echo t2m ;; orog) echo orography ;; lsm) echo land_sea_mask ;;
        tp) echo tp ;; 2r) echo relative_humidity ;; sp) echo surface_pressure ;;
        10si) echo wind_speed ;; 10wdir) echo wind_direction ;; *) return 1 ;;
    esac
}

# ---- run parameters: read from config; env vars (from the wrapper) override ----
DT=$(get_cfg data-types)
[ -n "$DT" ] || { echo "no --data-types in $CFG"; exit 1; }
N_TARGET=$(get_cfg n-target-data); N_TARGET="${N_TARGET:-1}"
REG=$(get_cfg region); REG="${REG:-ger}"
CFG_START=$(get_cfg start-year); CFG_END=$(get_cfg end-year)
START_YEAR="${START_YEAR:-$CFG_START}"
END_YEAR="${END_YEAR:-$CFG_END}"
[ -n "$START_YEAR" ] && [ -n "$END_YEAR" ] || { echo "no year span (config or env)"; exit 1; }

if [ -z "$INPUT_VARS" ]; then
    IFS=',' read -ra TYPES <<< "$DT"
    VARS=()
    for t in "${TYPES[@]}"; do
        v=$(type2var "$t") || { echo "unknown data type '$t' in $CFG"; exit 1; }
        VARS+=("$v")
    done
    n=${#VARS[@]}
    TARGET_VAR="${VARS[$((n-N_TARGET))]}"
    INPUT_VARS=$(printf '%s ' "${VARS[@]:0:$((n-N_TARGET))}")
fi
TARGET_VAR="${TARGET_VAR:-relative_humidity}"
STATIC_VARS="${STATIC_VARS:-orography}"
MODEL_NAMES="${MODEL_NAMES:-best}"

DATA_TPL=$(get_cfg data-file-template)
STATIC_TPL=$(get_cfg static-file-template)
TARGET_TPL=$(get_cfg target-file-template)
MASK_TPL=$(get_cfg mask-file-template)
STATIC_MASK_TPL=$(get_cfg static-mask-template)
STEADY_TPL=$(get_cfg steady-mask-file)
STATIC_YEAR=$(get_cfg static-year)

subst() { echo "$1" | sed "s/{year}/$2/g; s/{var}/$3/g; s/{reg}/$REG/g"; }

echo "================ slurm job ==================="
echo "job id      : ${SLURM_JOB_ID:-none}"
echo "node(s)     : ${SLURM_NODELIST:-localhost}"
echo "config      : $CFG"
echo "years       : $START_YEAR..$END_YEAR"
echo "INPUT_VARS  : $INPUT_VARS"
echo "TARGET_VAR  : $TARGET_VAR"
echo "STATIC_VARS : $STATIC_VARS"
echo "REG         : $REG"
echo "static year : $STATIC_YEAR"
echo "templates:"
echo "  data   : $DATA_TPL"
echo "  static : $STATIC_TPL"
echo "  target : $TARGET_TPL"
echo "  mask   : $MASK_TPL"
echo "  steady : $STEADY_TPL"
echo "---- config content ----"
cat "$CFG"
echo "---- end config ----"
echo "=============================================="

for ((year=START_YEAR; year<=END_YEAR; year++)); do
    DATA_NAMES=""; skip=0
    for v in $INPUT_VARS; do
        if [[ " $STATIC_VARS " == *" $v "* ]]; then f=$(subst "$STATIC_TPL" "$STATIC_YEAR" "$v")
        else f=$(subst "$DATA_TPL" "$year" "$v"); fi
        if [ ! -f "$f" ]; then echo "=== SKIP $year: missing $f ==="; skip=1; break; fi
        DATA_NAMES+="${f},"
    done
    TARGET_FILE=$(subst "$TARGET_TPL" "$year" "$TARGET_VAR")
    if [ $skip -eq 0 ] && [ ! -f "$TARGET_FILE" ]; then
        echo "=== SKIP $year: missing $TARGET_FILE ==="; skip=1
    fi
    [ $skip -eq 1 ] && continue
    DATA_NAMES+="${TARGET_FILE}"

    MASK_NAMES=""
    for v in $INPUT_VARS $TARGET_VAR; do
        if [[ " $STATIC_VARS " == *" $v "* ]]; then f=$(subst "$STATIC_MASK_TPL" "$STATIC_YEAR" "$v")
        else f=$(subst "$MASK_TPL" "$year" "$v"); fi
        if [ ! -f "$f" ]; then echo "=== SKIP $year: missing mask $f ==="; skip=1; break; fi
        MASK_NAMES+="${f},"
    done
    [ $skip -eq 1 ] && continue
    MASK_NAMES="${MASK_NAMES%,}"
    STEADY_MASK=$(subst "$STEADY_TPL" "$STATIC_YEAR" "$TARGET_VAR")

    N_TYPES=$(echo "$DT" | awk -F',' '{print NF}')
    N_NAMES=$(echo "$DATA_NAMES" | awk -F',' '{print NF}')
    if [ "$N_TYPES" -ne "$N_NAMES" ]; then
        echo "=== SKIP $year: $N_NAMES data names vs $N_TYPES data types in $CFG ==="
        continue
    fi

    echo "=== Processing year $year ==="
    echo "data-names : $DATA_NAMES"
    echo "mask-names : $MASK_NAMES"
    echo "steady     : $STEADY_MASK"
    echo "xrdss      : $(echo "$DATA_NAMES" | cut -d',' -f1)"
    python3 -m climatereconstructionai.evaluate \
        --load-from-file "$CFG" \
        --data-names "$DATA_NAMES" \
        --mask-names "$MASK_NAMES" \
        --steady-masks "$STEADY_MASK" \
        --xrdss-paths "$(echo "$DATA_NAMES" | cut -d',' -f1)" \
        --eval-names "cerra_dm_${TARGET_VAR}_${REG}_${year}_crai" \
        --model-names "$MODEL_NAMES"
    echo "=== Completed year $year ==="
done

echo "All years $START_YEAR..$END_YEAR processed"