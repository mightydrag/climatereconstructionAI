#!/bin/bash

#SBATCH -J crai-eval-t2m-ger
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --account=uo1075
#SBATCH --time=10:00:00
#SBATCH --mem=480G
#SBATCH --constraint a100_80
#SBATCH --mail-user=mats.militzer@studium.uni-hamburg.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions/input/slurm-scripts/eval_slurm_output/eval-cnn-t2m-ger.o%j
#SBATCH --error=/work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions/input/slurm-scripts/eval_slurm_output/eval-cnn-t2m-ger.e%j
source ~/.bashrc
mamba activate crai
cd /work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions/

# ----- config: adjust these to your actual file naming -----
BASENAME="cerra_dm_t2m_ger"
SUFFIX="_lowres_values_in_highres_grid"        # input files: <BASENAME>_<year><SUFFIX>.nc
TARGET_SUFFIX=""                        # ASSUMPTION: target files: <BASENAME>_<year>_target.nc
EXT=".nc"
START_YEAR=2016
END_YEAR=2025
DATA_DIR="/work/uo1075/u244085/cerra_data/t2m/"            # dir holding the data .nc files
MASK_DIR="/work/uo1075/u244085/cerra_data/mask/"
MASK_FILE="cerra_dm_t2m_ger_2025_mask.nc"                   # ASSUMPTION: per-timestep masks; must match data length per year
STEADY_MASK_FILE="cerra_dm_t2m_ger_2025_steady_mask.nc"
XR_TEMPLATE="/work/uo1075/u244085/cerra_data/t2m/cerra_dm_t2m_ger_${START_YEAR}${SUFFIX}.nc"  # any file on the high-res grid, containing the t2m variable
CONFIG_FILE="eval-cnn-cerra-t2m-ger_sort_en4_po1_batch.txt"
# ------------------------------------------------------------

echo '---Printing Config---'
cat ./input/levante/$CONFIG_FILE
echo '--- Config END---'

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
        --eval-names "cerra_dm_t2m_ger_${year}_crai"
    echo "=== Completed year $year ==="
done

echo "All years $START_YEAR..$END_YEAR processed"