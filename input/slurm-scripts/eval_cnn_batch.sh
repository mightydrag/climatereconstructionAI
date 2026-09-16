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
#SBATCH --output=/work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions/input/slurm-scripts/eval_slurm_output/eval-cnn.o%j
#SBATCH --error=/work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions/input/slurm-scripts/eval_slurm_output/eval-cnn.e%j

source ~/.bashrc
mamba activate crai
cd /work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions/

# ----- config: adjust these to your actual file naming -----
BASENAME="era5_land_t2max_daily"          # e.g. era5_land_t2max_daily_2021_europe_lowres_values_in_highres_grid.nc
SUFFIX="_europe_lowres_values_in_highres_grid"
EXT=".nc"
START_YEAR=2000
END_YEAR=2024
DATA_DIR="/work/uo1075/u244085/trees_in_the_city/data/era5_land/t2max/lowres_in_highres_grid"   # dir holding the data .nc files
MASK_FILE="/work/uo1075/u244085/trees_in_the_city/data/era5_land/t2max/mask/era5_land_t2max_daily_2000_europe_binary_mask.nc"
STEADY_MASK_FILE="/work/uo1075/u244085/trees_in_the_city/data/era5_land/t2max/mask/era5_land_t2max_daily_1950_europe_land_ocean_steady_mask.nc"
# ------------------------------------------------------------

for ((year=START_YEAR; year<=END_YEAR; year++)); do
    DATA_FILE="${BASENAME}_${year}${SUFFIX}${EXT}"
    FULL_PATH="${DATA_DIR}/${DATA_FILE}"

    if [ ! -f "$FULL_PATH" ]; then
        echo "=== SKIP $year: file not found -> $DATA_FILE ==="
        continue
    fi

    echo "=== Processing year $year: $DATA_FILE ==="

    python3 -m climatereconstructionai.evaluate \
        --load-from-file ./input/levante/eval-cnn-t2max_batch.txt \
        --data-names "$FULL_PATH" \
        --mask-names "$MASK_FILE" \
        --steady-masks "$STEADY_MASK_FILE" \
        --eval-names "era5_land_t2max_daily_${year}_europe_lowres_values_in_highres_grid"
    echo "=== Completed year $year ==="
done

echo "All years $START_YEAR..$END_YEAR processed"