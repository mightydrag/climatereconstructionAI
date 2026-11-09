#!/bin/bash

# Job name and log paths are passed on the sbatch command line by
# submit_eval.sh. All settings below have sensible defaults and can be
# overridden via environment variables (the wrapper does this with
# `sbatch --export=ALL,KEY=value,...`).
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --account=kd1418
#SBATCH --time=00:30:00
#SBATCH --mem=480G
#SBATCH --constraint a100_80
#SBATCH --mail-user=mats.militzer@studium.uni-hamburg.de
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/.bashrc
mamba activate crai
cd /work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions/

# ----- config: defaults; each can be overridden via environment -----------
CONFIG_FILE="${CONFIG_FILE:-eval-cnn-cerra-t2m-ger_sort_en4_po1_batch.txt}"

MODEL_NAMES="${MODEL_NAMES:-best}"
BASENAME="${BASENAME:-cerra_dm_t2m_ger}"
SUFFIX="${SUFFIX:-_lowres_values_in_highres_grid}"   # input files: <BASENAME>_<year><SUFFIX>.nc
TARGET_SUFFIX="${TARGET_SUFFIX:-}"                  # ASSUMPTION: target files: <BASENAME>_<year>_target.nc
EXT="${EXT:-.nc}"
START_YEAR="${START_YEAR:-2016}"
END_YEAR="${END_YEAR:-2025}"
DATA_DIR="${DATA_DIR:-/work/uo1075/u244085/cerra_data/t2m/}"   # dir holding the data .nc files
MASK_DIR="${MASK_DIR:-/work/uo1075/u244085/cerra_data/mask/}"
MASK_FILE="${MASK_FILE:-cerra_dm_t2m_ger_2025_mask.nc}"        # ASSUMPTION: per-timestep masks; must match data length per year
STEADY_MASK_FILE="${STEADY_MASK_FILE:-cerra_dm_t2m_ger_2025_steady_mask.nc}"
# built from the values above unless set explicitly:
XR_TEMPLATE="${XR_TEMPLATE:-${DATA_DIR}/${BASENAME}_${START_YEAR}${SUFFIX}.nc}"  # any file on the high-res grid, containing the t2m variable
# ---------------------------------------------------------------------------

echo '---Printing Config---'
cat ./input/levante/$CONFIG_FILE
echo '--- Config END---'

echo '---Settings---'
echo "BASENAME=${BASENAME}"
echo "SUFFIX=${SUFFIX}"
echo "TARGET_SUFFIX=${TARGET_SUFFIX}"
echo "START_YEAR=${START_YEAR}  END_YEAR=${END_YEAR}"
echo "DATA_DIR=${DATA_DIR}"
echo "MASK_FILE=${MASK_FILE}  STEADY_MASK_FILE=${STEADY_MASK_FILE}"
echo "XR_TEMPLATE=${XR_TEMPLATE}"
echo "MODEL_NAMES=${MODEL_NAMES}"
echo '--- Settings END ---'

for ((year=START_YEAR; year<=END_YEAR; year++)); do
    INPUT_FILE="${BASENAME}_${year}${SUFFIX}${EXT}"
    TARGET_FILE="${BASENAME}_${year}${TARGET_SUFFIX}${EXT}"
    INPUT_PATH="${DATA_DIR}/${INPUT_FILE}"
    TARGET_PATH="${DATA_DIR}/${TARGET_FILE}"

    if [ ! -f "$INPUT_PATH" ]; then
        echo "=== SKIP $year: input file not found -> $INPUT_FILE ==="
        continue
    fi
    if [ ! -f "$TARGET_PATH" ]; then
        echo "=== SKIP $year: target file not found -> $TARGET_FILE ==="
        continue
    fi

    echo "=== Processing year $year ==="

    python3 -m climatereconstructionai.evaluate \
        --load-from-file ./input/levante/$CONFIG_FILE \
        --data-names "${INPUT_PATH},${TARGET_PATH}" \
        --mask-names "${MASK_DIR}/${MASK_FILE},${MASK_DIR}/${MASK_FILE}" \
        --steady-masks "${MASK_DIR}/${STEADY_MASK_FILE}" \
        --xrdss-paths "${XR_TEMPLATE},${XR_TEMPLATE}" \
        --eval-names "cerra_dm_t2m_ger_${year}_crai" \
        --model-names "${MODEL_NAMES}"
    echo "=== Completed year $year ==="
done

echo "All years $START_YEAR..$END_YEAR processed"